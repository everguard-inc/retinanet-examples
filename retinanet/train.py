import os
from math import isfinite
from shutil import copyfile
from statistics import mean

import torch
from apex import amp
from apex.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from .backbones.layers import convert_fixedbn_model
from .dali import DaliDataIterator
from .data import DataIterator, RotatedDataIterator
from .infer import infer
from .scheduler import CosineAnnealingWarmUpWarmRestartsCooldown
from .utils import Profiler, ignore_sigint, post_metrics


def train(
    model,
    state,
    path,
    annotations,
    val_path,
    val_annotations,
    resize,
    max_size,
    jitter,
    batch_size,
    iterations,
    val_iterations,
    mixed_precision,
    lr,
    warmup,
    milestones,
    gamma,
    is_master=True,
    world=1,
    use_dali=True,
    verbose=True,
    metrics_url=None,
    logdir=None,
    rotate_augment=False,
    augment_brightness=0.0,
    augment_contrast=0.0,
    augment_hue=0.0,
    augment_saturation=0.0,
    regularization_l2=0.0001,
    rotated_bbox=False,
    absolute_angle=False,
):
    "Train the model on the given dataset"

    # Prepare model
    nn_model = model
    stride = model.stride

    model = convert_fixedbn_model(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # Setup optimizer and schedule
    optimizer = SGD(model.parameters(), lr=lr, weight_decay=regularization_l2, momentum=0.9)

    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level="O2" if mixed_precision else "O0",
        keep_batchnorm_fp32=True,
        loss_scale=128.0,
        verbosity=is_master,
    )

    if world > 1:
        model = DistributedDataParallel(model)
    model.train()

    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    # def schedule(train_iter):
    #     if warmup and train_iter <= warmup:
    #         return 0.9 * train_iter / warmup + 0.1
    #     return gamma ** len([m for m in milestones if m <= train_iter])

    # scheduler = LambdaLR(optimizer, schedule)
    scheduler = CosineAnnealingWarmUpWarmRestartsCooldown(optimizer, T_0=iterations, T_warmup=warmup, eta_max=lr)

    # Prepare dataset
    if verbose:
        print("Preparing dataset...")
    if rotated_bbox:
        if use_dali:
            raise NotImplementedError("This repo does not currently support DALI for rotated bbox detections.")
        data_iterator = RotatedDataIterator(
            path,
            jitter,
            max_size,
            batch_size,
            stride,
            world,
            annotations,
            training=True,
            rotate_augment=rotate_augment,
            augment_brightness=augment_brightness,
            augment_contrast=augment_contrast,
            augment_hue=augment_hue,
            augment_saturation=augment_saturation,
            absolute_angle=absolute_angle,
        )
    else:
        data_iterator = (DaliDataIterator if use_dali else DataIterator)(
            path,
            jitter,
            max_size,
            batch_size,
            stride,
            world,
            annotations,
            training=True,
            rotate_augment=rotate_augment,
            augment_brightness=augment_brightness,
            augment_contrast=augment_contrast,
            augment_hue=augment_hue,
            augment_saturation=augment_saturation,
        )
    if verbose:
        print(data_iterator)

    if verbose:
        print(
            "    device: {} {}".format(
                world, "cpu" if not torch.cuda.is_available() else "GPU" if world == 1 else "GPUs"
            )
        )
        print("     batch: {}, precision: {}".format(batch_size, "mixed" if mixed_precision else "full"))
        print(" BBOX type:", "rotated" if rotated_bbox else "axis aligned")
        print("Training model for {} iterations...".format(iterations))

    # Create TensorBoard writer
    if logdir is not None:
        logs_folder = os.path.join(logdir, "logs")
        os.makedirs(logs_folder, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter

        if is_master and verbose:
            print("Writing TensorBoard logs to: {}".format(logs_folder))
        writer = SummaryWriter(log_dir=logs_folder)

        checkpoints_folder = os.path.join(logdir, "checkpoints")
        os.makedirs(checkpoints_folder, exist_ok=True)
        top_3_scores = [0, 0, 0]

    profiler = Profiler(["train", "fw", "bw"])
    iteration = state.get("iteration", 0)
    while iteration < iterations:
        if logdir is not None:
            state["path"] = os.path.join(checkpoints_folder, "last.pth")
        cls_losses, box_losses = [], []
        for _, (data, target) in enumerate(data_iterator):
            if iteration >= iterations:
                break

            # Forward pass
            profiler.start("fw")

            optimizer.zero_grad()
            cls_loss, box_loss = model([data, target])
            del data
            profiler.stop("fw")

            # Backward pass
            profiler.start("bw")
            with amp.scale_loss(cls_loss + box_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            scheduler.step()

            # Reduce all losses
            cls_loss, box_loss = cls_loss.mean().clone(), box_loss.mean().clone()
            if world > 1:
                torch.distributed.all_reduce(cls_loss)
                torch.distributed.all_reduce(box_loss)
                cls_loss /= world
                box_loss /= world
            if is_master:
                cls_losses.append(cls_loss)
                box_losses.append(box_loss)

            if is_master and not isfinite(cls_loss + box_loss):
                raise RuntimeError("Loss is diverging!\n{}".format("Try lowering the learning rate."))

            del cls_loss, box_loss
            profiler.stop("bw")

            iteration += 1
            profiler.bump("train")
            if is_master and (profiler.totals["train"] > 60 or iteration == iterations):
                focal_loss = torch.stack(list(cls_losses)).mean().item()
                box_loss = torch.stack(list(box_losses)).mean().item()
                learning_rate = optimizer.param_groups[0]["lr"]
                if verbose:
                    msg = "[{:{len}}/{}]".format(iteration, iterations, len=len(str(iterations)))
                    msg += " focal loss: {:.3f}".format(focal_loss)
                    msg += ", box loss: {:.3f}".format(box_loss)
                    msg += ", {:.3f}s/{}-batch".format(profiler.means["train"], batch_size)
                    msg += " (fw: {:.3f}s, bw: {:.3f}s)".format(profiler.means["fw"], profiler.means["bw"])
                    msg += ", {:.1f} im/s".format(batch_size / profiler.means["train"])
                    msg += ", lr: {:.2g}".format(learning_rate)
                    print(msg, flush=True)

                if logdir is not None:
                    writer.add_scalar("focal_loss", focal_loss, iteration)
                    writer.add_scalar("box_loss", box_loss, iteration)
                    writer.add_scalar("learning_rate", learning_rate, iteration)
                    del box_loss, focal_loss

                if metrics_url:
                    post_metrics(
                        metrics_url,
                        {
                            "focal loss": mean(cls_losses),
                            "box loss": mean(box_losses),
                            "im_s": batch_size / profiler.means["train"],
                            "lr": learning_rate,
                        },
                    )

                # Save model weights
                if not val_annotations:
                    state.update(
                        {
                            "iteration": iteration,
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        }
                    )
                    with ignore_sigint():
                        nn_model.save(state)

                profiler.reset()
                del cls_losses[:], box_losses[:]

            if val_annotations and (iteration == iterations or iteration % val_iterations == 0):
                validation_final_metrics = infer(
                    model,
                    val_path,
                    None,
                    resize,
                    max_size,
                    batch_size,
                    annotations=val_annotations,
                    mixed_precision=mixed_precision,
                    is_master=is_master,
                    world=world,
                    use_dali=use_dali,
                    is_validation=True,
                    verbose=False,
                    rotated_bbox=rotated_bbox,
                )
                state.update(
                    {"iteration": iteration, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
                )
                with ignore_sigint():
                    nn_model.save(state)
                if validation_final_metrics is not None and logdir is not None:
                    # tensorboard write
                    for key, value in validation_final_metrics.items():
                        writer.add_scalar(key, value, iteration)

                    score = validation_final_metrics["total_infraction_f1"]
                    for index, top_score in enumerate(top_3_scores):
                        if score > top_score:
                            top_3_scores = top_3_scores[:index] + [score] + top_3_scores[index : len(top_3_scores) - 1]
                            for index_left in range(index + 1, len(top_3_scores))[::-1]:
                                src = os.path.join(checkpoints_folder, f"best_top_{index_left}.pth")
                                if os.path.exists(src):
                                    copyfile(
                                        src, os.path.join(checkpoints_folder, f"best_top_{index_left + 1}.pth"),
                                    )
                            state.update(
                                {
                                    "path": os.path.join(checkpoints_folder, f"best_top_{index + 1}.pth"),
                                    "score": score,
                                }
                            )
                            with ignore_sigint():
                                nn_model.save(state)
                            # return back
                            state.pop("score")
                            break
                model.train()

            if logdir is not None:
                writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], iteration)

            if (iteration == iterations and not rotated_bbox) or (iteration > iterations and rotated_bbox):
                break

    if logdir is not None:
        writer.close()

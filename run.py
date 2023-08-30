import os

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.runners import train_epoch, validate_epoch
from src.utils import load_model, masks_to_submission, rotate45

if __name__ == "__main__":
    # If a GPU is available
    if not torch.cuda.is_available():
        raise Exception(
            "Things will go much quicker if you enable a GPU in Colab under"
            " 'Runtime / Change Runtime Type'"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    foreground_threshold = (
        0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    )

    # assign a label to a patch

    dataset_train = []
    labels_train = []
    for x in (0, 1):
        for y in (-1, 1):
            for z in (-1, 1):
                dataset_train += [
                    mpimg.imread(f"training/images/satImage_{i:03}.png").transpose(2, x, 1)[
                        :, ::y, ::z
                    ]
                    for i in range(1, 101)
                ]
                dataset_train += [
                    rotate45(mpimg.imread(f"training/images/satImage_{i:03}.png")).transpose(
                        (2, x, 1)
                    )[:, ::y, ::z]
                    for i in range(1, 101)
                ]

                labels_train += [
                    mpimg.imread(f"training/groundtruth/satImage_{i:03}.png")
                    .round()
                    .astype(np.uint8)
                    .transpose(x == 0, x)[::y, ::z]
                    for i in range(1, 101)
                ]
                labels_train += [
                    rotate45(mpimg.imread(f"training/groundtruth/satImage_{i:03}.png"))
                    .round()
                    .astype(np.uint8)
                    .transpose(x == 0, x)[::y, ::z]
                    for i in range(1, 101)
                ]

    dataset_test = np.array(
        [mpimg.imread(f"test_set_images/test_{i}/test_{i}.png") for i in range(1, 51)]
    )
    dataset_test = np.moveaxis(dataset_test, -1, 1)
    dataset_test = torch.from_numpy(dataset_test.copy())

    random_seed = 0

    train_size = 0.95

    inputs_train, inputs_valid = train_test_split(
        dataset_train, random_state=random_seed, train_size=train_size, shuffle=True
    )

    targets_train, targets_valid = train_test_split(
        labels_train, random_state=random_seed, train_size=train_size, shuffle=True
    )

    inputs_train = [torch.from_numpy(img.copy()) for img in inputs_train]
    inputs_valid = [torch.from_numpy(img.copy()) for img in inputs_valid]
    targets_train = [torch.from_numpy(img.copy()).type(torch.LongTensor) for img in targets_train]
    targets_valid = [torch.from_numpy(img.copy()).type(torch.LongTensor) for img in targets_valid]

    # 1 image per batch actually
    dataset_training = [
        (inputs_train[i][None], targets_train[i][None]) for i in range(len(inputs_train))
    ]
    dataset_valid = [
        (inputs_valid[i][None], targets_valid[i][None]) for i in range(len(inputs_valid))
    ]

    os.makedirs("unet_states/Unet", exist_ok=True)

    # load model
    # set to 0 to start from scratch again or to 'latest' to continue training from saved checkpoint
    start_epoch = 0
    model, epoch = load_model(epoch=start_epoch)
    model = model.to(device)

    # optimizer
    try:
        learning_rate = np.load(f"unet_states/Unet/lr_epoch_{epoch}.npy")[0]
    except FileNotFoundError:
        learning_rate = np.load("unet_states/Unet/lr_epoch_0.npy")[0]  # 0.01
    momentum = 0.9
    gamma = 0.95
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optim, gamma=gamma)
    min_loss = torch.tensor(float("inf"))
    scheduler_counter = 0

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    num_epochs = epoch + 35

    # do epochs
    while epoch < num_epochs:
        # training
        model, loss_train, f1_train, oa_train = train_epoch(
            dataset_training, model, optim, device, criterion
        )

        # validation
        loss_val, f1_val, oa_val = validate_epoch(dataset_valid, model, device, criterion)

        # print stats
        str_output = (
            f"[Ep. {epoch + 1}/{num_epochs}] Loss train: {loss_train:.2f},"
            " Loss val: {loss_val:.2f}; F1 train: {100 * f1_train:.2f}, "
            "F1 val: {100 * f1_val:.2f}; OA train: {100 * oa_train:.2f},"
            " OA val: {100 * oa_val:.2f}"
        )
        print(str_output)

        # for LR scheduler
        scheduler_counter += 1
        is_best = loss_val < min_loss
        if is_best:
            scheduler_counter = 0
            min_loss = min(loss_val, min_loss)

        if scheduler_counter >= 2:
            lr_scheduler.step()
            scheduler_counter = 0
        # end for LR scheduler

        # save model
        epoch += 1

    imgs = []
    for img in dataset_test:
        with torch.no_grad():
            output = model.forward(img[None].to(device))
            yhat2 = output.argmax(dim=1).to(torch.float32)[0].cpu().numpy()
        imgs.append(yhat2)

    number_submission = np.load("submissions/number_submission.npy")[0]
    print(number_submission)

    submission_filename = f"submissions/submission_{number_submission}_BRAZ_DURAND_NICOLLE.csv"

    # actual best : 16 : F1 0.884
    masks_to_submission(submission_filename, imgs, foreground_threshold=foreground_threshold)
    number_submission += 1
    np.save("submissions/number_submission.npy", np.array([number_submission]))

    torch.cuda.empty_cache()

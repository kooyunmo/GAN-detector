import torch
import torch.nn as nn
import numpy as np
from models.Xception.xception import xception
from utils.args import parse_args
from utils.preprocess import preprocess

torch.backends.cudnn.benchmark = True


def test(model, test_loader, classes):
    num_classes = len(classes)
    test_loss = 0.0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))

    for inp, target in test_loader:
        inp, target = inp.cuda(), target.cuda()

        preds = model(inp)
        loss = criterion(preds, target)
        test_loss += loss.item * inp.size(0)

        # Convert probs to predicted class
        _, pred_class = torch.max(preds, 1)

        # Compare pred and true label
        correct_tensor = pred_class.eq(target.data.view_as(pred_class))
        correct = np.squeeze(correct_tensor.cpu().numpy())

        # Calculate test acc for each class
        for i in range(num_classes):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # Average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print("Test Loss: {:.6f}".format(test_loss))

    for i in range(num_classes):
        if class_total[i] > 0:
            print("Test Accuracy of {}: {:.4f} ({}/{})".format(classes[i], 100 * class_correct[i] / class_total[i],
                                                           np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print("Test Accuracy of {}: N/A (No training example)".format(classes[i]))
    
    print("\n" + "#" * 20)
    print("Final Test Accuracy: {:.4f} ({}/{})".format(100.0 * np.sum(class_correct) / np.sum(class_total),
                                                       np.sum(class_correct), np.sum(class_total)))
    print("#" * 20)


if __name__ == '__main__':
    args = parse_args()

    classes = ['stylegan', 'pggan', 'msgstylegan', 'vgan', 'real']

    train_loader, valid_loader, test_loader = preprocess(trainset_path=args.data_dir + "/train",
                                                         testset_path=args.data_dir + "/test",
                                                         classes=classes,
                                                         num_workers=40,
                                                         batch_size=args.batch_size,
                                                         validation_ratio=0.3)

    model = xception(pretrained=False)
    model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    test(model, test_loader, classes)

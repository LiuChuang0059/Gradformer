import csv
import os


def results_to_file(args, val, test):

    if not os.path.exists('./results/{}'.format(args.dataset)):
        print("=" * 20)
        print("Create Results File !!!")

        os.makedirs('./results/{}'.format(args.dataset))

    filename = "./results/{}/result.csv".format(
        args.dataset)

    headerList = ["Method", "Layer-Num", "Slope", "n_hop", "gamma", "drop_out", "attn_drop", "drop_path",
                  "::::::::", "val", "test"]

    with open(filename, "a+") as f:

        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        line = "{}, {}, {}, {}, {}, {}, {}, {}, :::::::::, {:.5f}, {:.5f}\n".format(
            args.model_type, args.num_layers, args.slope, args.n_hop, args.gamma, args.dropout,
            args.attn_dropout, args.drop_prob, val, test)
        f.write(line)

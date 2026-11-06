def solve(x):
    answer, offset = [], 0
    while True:
        t = offset + x["train_size"]
        v = t + x["label_horizon"] + x["embargo"]
        end_v = v + x["validation_size"]
        test = end_v + x["label_horizon"] + x["embargo"]
        end = test + x["test_size"]
        if end > len(x["sessions"]):
            break
        answer.append(
            {"train": x["sessions"][offset:t], "validation": x["sessions"][v:end_v], "test": x["sessions"][test:end]}
        )
        offset += x["test_size"]
    return {"folds": answer}

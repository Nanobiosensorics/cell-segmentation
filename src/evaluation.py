# Evaluates the test dataset by the provided methods (Watershed, etc) by the metrics (accuracy, execution time, etc).
def evaluate_dataset(test_dataloader, methods, metrics):
    result = {}

    for method in methods:
        method_name = method.__class__.__name__

        result[method_name] = {}
        for metric in metrics:
            metric_name = metric.__name__
            result[method_name][metric_name] = 0

        for test_imgs, test_markers in test_dataloader:
            test_predicts, execution_time = method.predict_batch(test_imgs)

            for marker, prediction in zip(test_markers, test_predicts):
                for metric in metrics:
                    metric_name = metric.__name__
                    result[method_name][metric_name] += metric(marker, prediction, execution_time / len(test_markers))
            
        for metric in metrics:
            metric_name = metric.__name__
            result[method_name][metric_name] /= len(test_dataloader.dataset)
    
    return result


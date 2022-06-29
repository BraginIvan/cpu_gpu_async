# torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./architectures/model.py --serialized-file data/resnet18.pth --handler handlers/my_handler.py --extra-files ./data/index_to_name.json
# torchserve --start --model-store model_store
# curl -X POST "localhost:8081/models?model_name=resnet-18&url=resnet-18.mar&batch_size=16&max_batch_delay=50&initial_workers=3&synchronous=true"
# curl -X DELETE http://localhost:8081/models/resnet-18/1.0

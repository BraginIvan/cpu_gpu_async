# Настройка окружения

- Создать conda env с python 3.9
- conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
- pip install -r requirements.txt


# Подготовка данных
### Скачать веса моделей
Скачать https://download.pytorch.org/models/resnet18-f37072fd.pth и положить в data/resnet18.pth
Скачать https://download.pytorch.org/models/resnet152-394f9c45.pth и положить в data/resnet152.pth
### Создать дирректорию с картинками
Достаточно скачать 5-6 разных картинок и положить в новую директорию. 

# Запуск torchserve тестов
Создать директорию для моделей
```
mkdir model_store
```
### Resnet-18
В одном окне запускаем 
```
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./utils/resnet18.py --serialized-file data/resnet18.pth --handler utils/my_handler.py --extra-files ./data/index_to_name.json
mv resnet-18.mar model_store
torchserve --stop
torchserve --start --model-store model_store
curl -X DELETE http://localhost:8081/models/resnet/1.0
curl -X POST "localhost:8081/models?model_name=resnet&url=resnet-18.mar&batch_size=32&max_batch_delay=50&initial_workers=3&synchronous=true"
```
В другом 
```
python send_requests.py --ports 1 --batch-size 1 --imgs-path /home/ivan/tmp --images-n 1600
```
### Resnet-152
В одном окне запускаем 
```
torch-model-archiver --model-name resnet-152 --version 1.0 --model-file ./utils/resnet152.py --serialized-file data/resnet152.pth --handler utils/my_handler.py --extra-files ./data/index_to_name.json
mv resnet-152.mar model_store
torchserve --stop
torchserve --start --model-store model_store
curl -X DELETE http://localhost:8081/models/resnet/1.0
curl -X POST "localhost:8081/models?model_name=resnet&url=resnet-152.mar&batch_size=16&max_batch_delay=50&initial_workers=3&synchronous=true"
```
В другом 
```
python send_requests.py --ports 1 --batch-size 1 --imgs-path /home/ivan/tmp --images-n 1608
```

Если появится ошибка что модель уже зарегистрированна

```
curl -X DELETE http://localhost:8081/models/resnet/1.0
```

Для освобождения портов 
```
torchserve --stop
```


# Запуск теста Синхронно в 1 процессе.
### Resnet-18
В одном окне запускаем 
```
python api_sync.py --model resnet18
```
В другом 
```
python send_requests.py --ports 1 --batch-size 80 --imgs-path /home/ivan/tmp --images-n 1600
```
### Resnet-152
В одном окне запускаем 
```
python api_sync.py --model resnet152
```
В другом 
```
python send_requests.py --ports 1 --batch-size 24 --imgs-path /home/ivan/tmp --images-n 1608
```


# Запуск теста Асинхронно. Несколько CPU воркеров
У меня не запустился этот тест на новом venv, зависает на model.to('cuda'), раньше всё работало.
### Resnet-18
В одном окне запускаем 
```
python api_async.py --model resnet18 
```
В другом 
```
python send_requests.py --ports 1 --batch-size 80 --imgs-path /home/ivan/tmp --images-n 1600
```
### Resnet-152
В одном окне запускаем 
```
python api_async.py --model resnet152
```
В другом 
```
python send_requests.py --ports 1 --batch-size 24 --imgs-path /home/ivan/tmp --images-n 1608
```


# Запуск теста Асинхронно. REST API на каждом воркере
### Resnet-18
В одном окне запускаем 
```
python api_ports.py --model resnet18 --ports 3 
```
В другом 
```
python send_requests.py --ports 3 --batch-size 80 --imgs-path /home/ivan/tmp --images-n 1600
```

### Resnet-152
В одном окне запускаем 
```
python api_ports_sync.py --model resnet152
```
В другом 
```
python send_requests.py --ports 3 --batch-size 24 --imgs-path /home/ivan/tmp --images-n 1608
```



# Запуск теста Сокеты вместо очередей
### Resnet-18
В одном окне запускаем 
```
python api_socket.py --model resnet18 --ports 3 
```
В другом 
```
python send_requests.py --ports 3 --batch-size 80 --imgs-path /home/ivan/tmp --images-n 1600
```
### Resnet-152
В одном окне запускаем 
```
python api_socket.py --model resnet152 --ports 3 
```
В другом 
```
python send_requests.py --ports 3 --batch-size 24 --imgs-path /home/ivan/tmp --images-n 1608
```
from data_ops import get_data
from model_ops import train_models, test_models
from models import get_models
from plots import create_plots

models = get_models()
train_data, test_data = get_data()
histories = train_models(models, train_data)

print(test_models(models, test_data))

create_plots(histories, 'accuracy', 'Train accuracy')
create_plots(histories, 'val_accuracy', 'Val accuracy')
create_plots(histories, 'loss', 'Train loss')
create_plots(histories, 'val_loss', 'Val loss')

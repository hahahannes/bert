# load cleaned data
# tokenize data
# train model
# -> model independent -> no bert specific code here e.g. text-to-tokens
# wandb logging 

class Pipeline():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def train_batch(self, batch):
        model_input = self.tokenizer(batch) 
        output = self.model(model_input)

    def train(self):
        for epoch in range(self.n_epochs):
            self.train_batch(batch)
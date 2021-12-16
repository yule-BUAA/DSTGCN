import torch.nn as nn


class fully_connected_layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize fullyConnectedNet.

        Parameters
        ----------
        input_size – The number of expected features in the input x  -> scalar
        hidden_size – The numbers of features in the hidden layer h  -> list
        output_size  – The number of expected features in the output x  -> scalar

        input -> (batch, in_features)

        :return
        output -> (batch, out_features)

        """
        super(fully_connected_layer, self).__init__()

        self.input_size = input_size
        # list
        self.hidden_size = hidden_size
        self.output_size = output_size
        fcList = []
        reluList = []
        for index in range(len(self.hidden_size)):
            if index != 0:
                input_size = self.hidden_size[index - 1]
            fc = nn.Linear(input_size, self.hidden_size[index])
            setattr(self, f'fc{index}', fc)
            fcList.append(fc)
            relu = nn.ReLU()
            setattr(self, f'relu{index}', relu)
            reluList.append(relu)
        self.last_fc = nn.Linear(self.hidden_size[-1], self.output_size)

        self.fcList = nn.ModuleList(fcList)
        self.reluList = nn.ModuleList(reluList)

    def forward(self, input_tensor):

        """
        :param input_tensor:
            2-D Tensor  (batch, input_size)

        :return:
            2-D Tensor (batch, output_size)
            output_tensor
        """
        for idx in range(len(self.fcList)):
            out = self.fcList[idx](input_tensor)
            out = self.reluList[idx](out)
            input_tensor = out
        # (batch, output_size)
        output_tensor = self.last_fc(input_tensor)

        return output_tensor

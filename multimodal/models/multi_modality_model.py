import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers, bidirectional=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, bidirectional =bidirectional)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x) # [seq_len, batch, hidden size]
        return out


class Text_AVG(nn.Module):
    def __init__(self, name = 'avg'):
        """
        The average operation is done before feed it to network due to memory considerations
        """
        super(Text_AVG, self).__init__()
        self.name = name

    def forward(self,x):
        return x


class Text_CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_heights, embedding_length, name ='cnn'):
        super(Text_CNN, self).__init__()
        
        """
        Arguments
        ---------
        in_channels : Number of input channels. Here it is 1 as the input data has dimension = (batch_size, num_seq, embedding_length)
        out_channels : Number of output channels after convolution operation performed on the input matrix
        kernel_heights : A list consisting of 3 different kernel_heights. Convolution will be performed 3 times and finally results from each kernel_height will be concatenated.
        embedding_length : Embedding dimension of GloVe word embeddings
        --------
        """
        
        self.name = name
        self.embedding_length = embedding_length
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_heights = kernel_heights
        self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], embedding_length), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], embedding_length), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], embedding_length), stride=1, padding=0)
        
    def conv_block(self, x, conv_layer):
        conv_out = conv_layer(x)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim)
        max_out = F.max_pool1d(activation, activation.shape[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
        return max_out
    
    def forward(self, x):
        # x shape [batch_size, num_doc, sen_len, emb_dim]
        res = []
        for i in range(x.shape[1]):
            # i is the number of docs in this batch 
            # input.size() = (batch_size, num_seq, embedding_length)
            x_i = x[:,i,:,:].unsqueeze(1) #(batch_size, 1, len_seq, embedding_length)
            max_out1 = self.conv_block(x_i, self.conv1) # [batch_size, out_channel]
            max_out2 = self.conv_block(x_i, self.conv2)
            max_out3 = self.conv_block(x_i, self.conv3)
            text_features = torch.cat((max_out1, max_out2, max_out3), 1) # [batch_size, out_channel*3]
            res.append(text_features)  
        res = torch.stack(res, dim =1) #[batch_size, num_doc, 768]
        return res


class Waveform_Pretrained(nn.Module):
    def __init__(self):
        super(Waveform_Pretrained, self).__init__()

    def forward(self, x):
        return x


class Waveform_CNN(nn.Module):
    def __init__(self, signal_length = 625):
        super(Waveform_CNN, self).__init__()
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        filters = 256
        self.maxpool = torch.nn.MaxPool1d(kernel_size=64, stride=4)
        self.cnn = torch.nn.Conv1d(in_channels=1, 
                                   out_channels=filters, 
                                   kernel_size=32,
                                    stride=4)
        self.first_conv = torch.nn.Sequential(
            self.cnn,
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.6)
        )
        strides = 2
        self.seq1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filters, out_channels=filters,
                            kernel_size=32, 
                            stride=2),
            torch.nn.BatchNorm1d(num_features=filters),
            torch.nn.ReLU(inplace=True),
        )
        self.seq2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filters, 
                            out_channels=filters, 
                            kernel_size=64, 
                            stride=strides),
            torch.nn.BatchNorm1d(num_features=filters),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.6)
        )
        self.seq3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=64, stride=strides),
            torch.nn.BatchNorm1d(num_features=filters),
            torch.nn.ReLU(inplace=True),
        )
        self.seq4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=64, stride=2),
            torch.nn.BatchNorm1d(num_features=filters),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5)
        )

        self.downsample = torch.nn.Linear(filters*7, 50)

    def forward(self, x):
        B, _, L = x.size()
        x = self.first_conv(x)
        x = self.seq1(x)
        x = self.seq2(x)
        x = self.seq3(x)
        x = x.view(B, -1)
        x = self.downsample(x)
        return x   #Logits


def FeatureSpreadWTime(features_i, weight_i):
    """
    args: features_i: shape of : (n, text feature dim) or (n, waveform feature dim)
          weight_i: shape of : (t, n) n is the number of docs or num of waveforms
    return: sentence_features at all timestamps
    """
    A = features_i.permute(1,0) #[768, n]
    B = weight_i.unsqueeze(1)
    C = A*B
    C = torch.sum(C, dim=2)
    # here it should be normalized by the num of non-zero elements, but in my experiments, the following normalization works better
    C/= A.shape[1]
    return C


def mean_text(sentence_features, dim):
    return torch.mean(sentence_features, dim = dim, keepdim = True)


class MultiModal_Model_decom_ts(nn.Module):
    def __init__(self, ts_model, decom_classes=2 ):
        super(MultiModal_Model_decom_ts, self).__init__()
        self.ts_model = ts_model
        self.decom_classes = decom_classes
        tsout_dim = self.ts_model.hidden_dim
        self.fc_decom = nn.Sequential(
                nn.Linear(tsout_dim, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.decom_classes))
        
    def forward(self, ts):
        batch_size = ts.shape[1]
        t = ts.shape[0]
        ts_output = self.ts_model(ts).float() # shape (seq_len, batch_size, num_directions* hidden_size)
        ts_output = ts_output.permute(1,0,2) #[b,t, 256]
        ts_output = ts_output.reshape(-1, ts_output.shape[-1])#[b*t, 256]
        decom_out = self.fc_decom(ts_output) #[b*t, 2]
        return decom_out


class MultiModal_Model_ihm_ts(nn.Module):
    def __init__(self, ts_model, ihm_classes=2):
        super(MultiModal_Model_ihm_ts, self).__init__()
        self.ts_model = ts_model
        self.ihm_classes = ihm_classes
        tsout_dim = self.ts_model.hidden_dim
        self.fc_decom = nn.Sequential(
            nn.Linear(tsout_dim, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.ihm_classes))

    def forward(self, ts):
        batch_size = ts.shape[1]
        t = ts.shape[0]
        ts_output = self.ts_model(ts).float()  # shape (seq_len, batch_size, num_directions* hidden_size)
        ts_output = ts_output.permute(1, 0, 2)  # [b,t, 256]
        ts_output = ts_output[:, 47, :] #[b, 1, 256]
        #ts_output = ts_output.reshape(-1, ts_output.shape[-1])  # [b*t, 256]
        decom_out = self.fc_decom(ts_output)  # [b, 1, 2]
        return decom_out


class MultiModal_Model_ihm_ts_wf(nn.Module):
    def __init__(self, ts_model, waveform_model, decom_classes=2):
        super(MultiModal_Model_ihm_ts_wf, self).__init__()
        self.ts_model = ts_model
        self.waveform_model = waveform_model
        self.decom_classes = decom_classes
        tsout_dim = self.ts_model.hidden_dim
        waveformout_dim = 100

        self.fc_decom = nn.Sequential(
            nn.Linear(tsout_dim + waveformout_dim, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.decom_classes))

    def timespread(self, waveform_out, ts_out, weight_mat=None):
        # waveform out [b, n, 100] n is the num of wfs
        # ts out [t, b, 256]
        time = ts_out.shape[0]
        res = []

        for i in range(waveform_out.shape[0]):
            ecg_feat = waveform_out[i, :, :]
            mask = weight_mat[i, :, :]
            new_feat = FeatureSpreadWTime(ecg_feat, mask)
            res.append(new_feat)

        res = torch.stack(res, dim=0)
        return res

    def forward(self, ts, waveform, waveform_weight_mat=None):
        batch_size = ts.shape[1]
        t = ts.shape[0]

        # waveform [b * n* 100]
        # print('waveform input shape', waveform.shape)
        # waveform_output = []
        # for i in range(waveform.shape[1]):
        #     #tmp = self.waveform_model(waveform[:,i,:].unsqueeze(1)) # [b, 100]
        #     tmp = self.waveform_model(waveform[:,i,:])
        #     waveform_output.append(tmp)
        # waveform_output = torch.stack(waveform_output, dim = 1)
        # print('waveform output shape', waveform_output.shape)

        ts_output = self.ts_model(ts)  # shape (seq_len, batch_size, num_directions* hidden_size)

        waveform_output = self.timespread(waveform, ts_output, waveform_weight_mat).float()
        ts_output = ts_output.permute(1,0, 2) #[batch, seq_len, hidden_size]
        #waveform_output = waveform_output.reshape(-1, waveform_output.shape[-1])
        #ts_output = ts_output.reshape(-1, ts_output.shape[-1])  # [b*t, 256]

        out = torch.cat((ts_output, waveform_output), dim=2)# shape [batch, seq_len, waveform_dim + ts_dim ]
        out = out[:, 47, :] # select only for last timepoint

        decom_out = self.fc_decom(out)  # [b*t, 2]
        return decom_out


class MultiModal_Model_ihm_text_ts_wf(nn.Module):
    def __init__(self, text_model, ts_model, waveform_model, ihm_classes=2):
        super(MultiModal_Model_ihm_text_ts_wf, self).__init__()
        self.text_model = text_model
        self.ts_model = ts_model
        self.waveform_model = waveform_model
        self.ihm_classes = ihm_classes
        tsout_dim = self.ts_model.hidden_dim
        waveformout_dim = 100

        self.fc_decom = nn.Sequential(
            nn.Linear(tsout_dim + waveformout_dim + 300, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.ihm_classes))

    def timespread(self, waveform_out, ts_out, weight_mat=None):
        # waveform out [b, n, 100] n is the num of wfs
        # ts out [t, b, 256]
        time = ts_out.shape[0]
        res = []

        for i in range(waveform_out.shape[0]):
            ecg_feat = waveform_out[i, :, :]
            mask = weight_mat[i, :, :]
            new_feat = FeatureSpreadWTime(ecg_feat, mask)
            res.append(new_feat)

        res = torch.stack(res, dim=0)
        return res

    def forward(self, text, text_weight_mat, ts, waveform, waveform_weight_mat=None):
        batch_size = ts.shape[1]
        t = ts.shape[0]
       # print('waveform output shape', waveform_output.shape)
        text_out = self.text_model(text)
        ts_output = self.ts_model(ts)  # shape (seq_len, batch_size, num_directions* hidden_size)
        text_out = self.timespread(text_out, ts_output, text_weight_mat).float()
        waveform_output = self.timespread(waveform, ts_output, waveform_weight_mat).float()
        ts_output = ts_output.permute(1, 0, 2)  # [batch, seq_len, hidden_size]
        # waveform_output = waveform_output.reshape(-1, waveform_output.shape[-1])
        # ts_output = ts_output.reshape(-1, ts_output.shape[-1])  # [b*t, 256]

        out = torch.cat((ts_output, waveform_output, text_out), dim=2)  # shape [batch, seq_len, waveform_dim + ts_dim ]
        out = out[:, 47, :]  # select only for last timepoint

        decom_out = self.fc_decom(out)  # [b*t, 2]
        return decom_out

class MultiModal_Model_ihm_text_ts(nn.Module):
    def __init__(self, text_model, ts_model, ihm_classes=2):
        super(MultiModal_Model_ihm_text_ts, self).__init__()
        self.text_model = text_model
        self.ts_model = ts_model
        self.ihm_classes = ihm_classes
        tsout_dim = self.ts_model.hidden_dim

        self.fc_decom = nn.Sequential(
            nn.Linear(tsout_dim + 300, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, self.ihm_classes))

    def timespread(self, waveform_out, ts_out, weight_mat=None):
        # waveform out [b, n, 100] n is the num of wfs
        # ts out [t, b, 256]
        time = ts_out.shape[0]
        res = []

        for i in range(waveform_out.shape[0]):
            ecg_feat = waveform_out[i, :, :]
            mask = weight_mat[i, :, :]
            new_feat = FeatureSpreadWTime(ecg_feat, mask)
            res.append(new_feat)

        res = torch.stack(res, dim=0)
        return res

    def forward(self, text, text_weight_mat, ts):
        batch_size = ts.shape[1]
        t = ts.shape[0]
       # print('waveform output shape', waveform_output.shape)
        text_out = self.text_model(text)
        ts_output = self.ts_model(ts)  # shape (seq_len, batch_size, num_directions* hidden_size)
        text_out = self.timespread(text_out, ts_output, text_weight_mat).float()
        ts_output = ts_output.permute(1, 0, 2)  # [batch, seq_len, hidden_size]
        # ts_output = ts_output.reshape(-1, ts_output.shape[-1])  # [b*t, 256]

        out = torch.cat((ts_output, text_out), dim=2)  # shape [batch, seq_len, waveform_dim + ts_dim ]
        out = out[:, 47, :]  # select only for last timepoint

        decom_out = self.fc_decom(out)  # [b*t, 2]
        return decom_out
class MultiModal_Model_decom_text_ts(nn.Module):
    def __init__(self, text_model, ts_model, decom_classes=2):
        super(MultiModal_Model_decom_text_ts, self).__init__()
        self.text_model = text_model
        self.ts_model = ts_model
        self.decom_classes = decom_classes
        tsout_dim = self.ts_model.hidden_dim
        textout_dim = 200
       
        self.fc_decom = nn.Sequential(
                nn.Linear(textout_dim+tsout_dim, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.decom_classes))
    
    def forward(self, texts, ts):
        batch_size = ts.shape[1]
        t = ts.shape[0]
        text_output = texts.reshape(-1, texts.shape[-1])  #[b*t, 200]
        ts_output = self.ts_model(ts) # shape (seq_len, batch_size, num_directions* hidden_size)
        ts_output = ts_output.reshape(-1, ts_output.shape[-1])#[b*t, 256]
        out = torch.cat((ts_output, text_output), dim =1)    
        decom_out = self.fc_decom(out) #[b*t, 2]
        return decom_out


class MultiModal_Model_decom_text_ts_wf(nn.Module):
    def __init__(self, text_model, ts_model, waveform_model, decom_classes=2):
        super(MultiModal_Model_decom_text_ts_wf, self).__init__()
        self.text_model = text_model
        self.ts_model = ts_model
        self.waveform_model = waveform_model
        self.decom_classes = decom_classes
        tsout_dim = self.ts_model.hidden_dim
        textout_dim = 200
        waveformout_dim = 100
       
        self.fc_decom = nn.Sequential(
                nn.Linear(textout_dim+tsout_dim+waveformout_dim, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.decom_classes))
        
    
    def timespread(self, waveform_out, ts_out, weight_mat = None):
        # waveform out [b, n, 200] n is the num of docs
        # ts out [t, b, 256]
        time = ts_out.shape[0]
        res = []
        
        for i in range(waveform_out.shape[0]):
            ecg_feat = waveform_out[i,:,:]
            mask = weight_mat[i,:,:]
            new_feat = FeatureSpreadWTime(ecg_feat, mask)
            res.append(new_feat)

        res = torch.stack(res, dim=0)
        return res

    
    
    def forward(self, texts, ts, waveform, waveform_weight_mat = None):
        batch_size = ts.shape[1]
        t = ts.shape[0]
        text_output = texts.reshape(-1, texts.shape[-1])#[b*t, 200]
        #print(text_output.shape)
        waveform_output = []
        for i in range(waveform.shape[1]):
            #tmp = self.waveform_model(waveform[:,i,:].unsqueeze(1)) # [b, 100]
            tmp = self.waveform_model(waveform[:,i,:])
            waveform_output.append(tmp)
        waveform_output = torch.stack(waveform_output, dim = 1)

        ts_output = self.ts_model(ts) # shape (seq_len, batch_size, num_directions* hidden_size)
        waveform_output = self.timespread(waveform_output, ts_output, waveform_weight_mat).float()
        waveform_output = waveform_output.reshape(-1, waveform_output.shape[-1])
        ts_output = ts_output.reshape(-1, ts_output.shape[-1])#[b*t, 256]

        out = torch.cat((text_output, ts_output, waveform_output), dim =1)
            
        decom_out = self.fc_decom(out) #[b*t, 2]
        return decom_out

class MultiModal_Model_decom_ts_wf(nn.Module):
    def __init__(self, ts_model, waveform_model, decom_classes=2):
        super(MultiModal_Model_decom_ts_wf, self).__init__()
        self.ts_model = ts_model
        self.waveform_model = waveform_model
        self.decom_classes = decom_classes
        tsout_dim = self.ts_model.hidden_dim
        waveformout_dim = 100
       
        self.fc_decom = nn.Sequential(
                nn.Linear(tsout_dim+waveformout_dim, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.decom_classes))
        
    
    def timespread(self, waveform_out, ts_out, weight_mat = None):
        # waveform out [b, n, 100] n is the num of wfs
        # ts out [t, b, 256]
        time = ts_out.shape[0]
        res = []
        
        for i in range(waveform_out.shape[0]):
            ecg_feat = waveform_out[i,:,:]
            mask = weight_mat[i,:,:]
            new_feat = FeatureSpreadWTime(ecg_feat, mask)
            res.append(new_feat)

        res = torch.stack(res, dim=0)
        return res

    
    
    def forward(self, ts, waveform, waveform_weight_mat = None):
        batch_size = ts.shape[1]
        t = ts.shape[0]
        
        # waveform [b * n* 100]
        # print('waveform input shape', waveform.shape)
        # waveform_output = []
        # for i in range(waveform.shape[1]):
        #     #tmp = self.waveform_model(waveform[:,i,:].unsqueeze(1)) # [b, 100]
        #     tmp = self.waveform_model(waveform[:,i,:])
        #     waveform_output.append(tmp)
        # waveform_output = torch.stack(waveform_output, dim = 1)
        # print('waveform output shape', waveform_output.shape)

        
        
        ts_output = self.ts_model(ts) # shape (seq_len, batch_size, num_directions* hidden_size)
        waveform_output = self.timespread(waveform, ts_output, waveform_weight_mat).float()
        waveform_output = waveform_output.reshape(-1, waveform_output.shape[-1])
        ts_output = ts_output.reshape(-1, ts_output.shape[-1])#[b*t, 256]
        


        out = torch.cat((ts_output, waveform_output), dim =1)
            
        decom_out = self.fc_decom(out) #[b*t, 2]
        return decom_out

# class MultiModal_Model_los(MultiModal_Model_decom):
#     def __init__(self, cnn_model, lstm_model,  tabs = False, text_postprocess = 'weighted_combine', los_classes=10):
#         super(MultiModal_Model_los, self).__init__(cnn_model = cnn_model, lstm_model = lstm_model)
#         self.cnn_model = cnn_model
#         self.lstm_model = lstm_model
#         self.text_postprocess = text_postprocess
#         self.los_classes = los_classes
#         self.tabs = tabs
#         if not self.tabs:
#             self.fc_los = nn.Linear(1024, self.los_classes)
#         else:
#             self.tabs_net = nn.Sequential(nn.Linear(13, 5), nn.ReLU())
#             self.fc_los = nn.Linear(1029, self.los_classes)
    
    
#     def forward(self,text, ts, weight_mat =None, tabs = None):
#         batch_size = ts.shape[1]
#         t = ts.shape[0]
#         if self.tabs:
#             tabs = tabs.repeat(1, t, 1)
#             tabs = self.tabs_net(tabs).float()
#             tabs = tabs.reshape(-1, tabs.shape[-1])

        
#         cnn_output = self.cnn_model(text)
#         rnn_output = self.lstm_model(ts).float() # shape (seq_len, batch_size, num_directions* hidden_size)
#         cnn_output = self.timespread(cnn_output, rnn_output, weight_mat).float()
#         rnn_output = rnn_output.permute(1,0,2) #[b,t, 256]

#         rnn_output = rnn_output.reshape(-1, rnn_output.shape[-1])#[b*t, 256]
#         cnn_output = cnn_output.reshape(-1, cnn_output.shape[-1]) #[b*t, 768]
#         if not self.tabs:
#             out = torch.cat((rnn_output, cnn_output), dim =1)
#         else:
#             out = torch.cat((rnn_output, cnn_output, tabs), dim =1)

#         los_out = self.fc_los(out) #[b*t, 10]
#         return los_out


# class MultiModal_Model_ihm(MultiModal_Model_decom):
#     def __init__(self, cnn_model, lstm_model,tabs = False, text_postprocess = 'weighted_combine', ihm_classes=2):
#         super(MultiModal_Model_ihm, self).__init__(cnn_model = cnn_model, lstm_model = lstm_model)
#         self.cnn_model = cnn_model
#         self.lstm_model = lstm_model
#         self.text_postprocess = text_postprocess
#         self.ihm_classes = ihm_classes
#         self.tabs = tabs
#         if not self.tabs:
#             self.fc_ihm = nn.Linear(1024, self.ihm_classes)
#         else:
#             self.tabs_net = nn.Sequential(nn.Linear(13, 5), nn.ReLU())
#             self.fc_ihm = nn.Linear(1029, self.ihm_classes)
    
    
#     def forward(self,text, ts, weight_mat =None, tabs = None):
#         batch_size = ts.shape[1]
#         t = ts.shape[0]

#         if self.tabs:
#             tabs = tabs.repeat(1, t, 1)
#             tabs = self.tabs_net(tabs).float()
#             tabs = tabs.reshape(-1, tabs.shape[-1])

#         cnn_output = self.cnn_model(text)
        
#         rnn_output = self.lstm_model(ts).float() # shape (seq_len, batch_size, num_directions* hidden_size)
#         cnn_output = self.timespread(cnn_output, rnn_output, weight_mat).float() 
#         rnn_output = rnn_output.permute(1,0,2) #[b,t, 256]
#         rnn_output = rnn_output.reshape(-1, rnn_output.shape[-1])#[b*t, 256]
#         cnn_output = cnn_output.reshape(-1, cnn_output.shape[-1]) #[b*t, 768]
        
#         if not self.tabs:
#             out = torch.cat((rnn_output, cnn_output), dim =1)
#         else:
#             out = torch.cat((rnn_output, cnn_output, tabs), dim =1)
        
#         out_copy = out.reshape(batch_size, -1, out.shape[1]) # [b, t, 1024]
#         ihm_out = out_copy[:, 48, :] #[b, 1024]
#         ihm_out = self.fc_ihm(ihm_out)
     
#         return ihm_out
        


class MultiModal_Model_3_task(nn.Module):
    def __init__(self, text_model, ts_model,tabs = False,  ihm_classes=2,\
         decom_classes=2, los_classes=10):
        super(MultiModal_Model_3_task, self).__init__()
        self.text_model = text_model
        self.ts_model = ts_model
        self.tabs = tabs
        self.decom_classes = decom_classes
        self.los_classes = los_classes
        self.ihm_classes = ihm_classes
        tsout_dim = self.ts_model.hidden_dim

        if self.text_model.name == 'cnn':
            textout_dim = 768
        elif self.text_model.name == 'avg':
            textout_dim = 200


        if not self.tabs:
            self.fc_decom = nn.Sequential(
                nn.Linear(tsout_dim+textout_dim, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.decom_classes))
            self.fc_los = nn.Sequential(
                nn.Linear(tsout_dim+textout_dim, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.los_classes))
            self.fc_ihm = nn.Sequential(
                nn.Linear(tsout_dim+textout_dim, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.ihm_classes))
        else:
            self.tabs_net = nn.Sequential(nn.Linear(13, 5), nn.ReLU())
            self.fc_decom = nn.Sequential(
                nn.Linear(tsout_dim+textout_dim+5, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.decom_classes))
            self.fc_los = nn.Sequential(
                nn.Linear(tsout_dim+textout_dim+5, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.los_classes))
            self.fc_ihm = nn.Sequential(
                nn.Linear(tsout_dim+textout_dim+5, 256),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(256, self.ihm_classes))

    def timespread(self, waveform_out, ts_out, weight_mat = None):
        # waveform out [b, n, 200] n is the num of docs
        # ts out [t, b, 256]
        time = ts_out.shape[0]
        res = []
        
        for i in range(waveform_out.shape[0]):
            ecg_feat = waveform_out[i,:,:]
            mask = weight_mat[i,:,:]
            new_feat = FeatureSpreadWTime(ecg_feat, mask)
            res.append(new_feat)

        res = torch.stack(res, dim=0)
        return res
    

    def forward(self,text, ts, weight_mat =None, tabs = None):
        batch_size = ts.shape[1]
        t = ts.shape[0]

        if self.tabs:
            tabs = tabs.repeat(1, t, 1)
            tabs = self.tabs_net(tabs).float()
            tabs = tabs.reshape(-1, tabs.shape[-1])

        ts_output = self.ts_model(ts).float() # shape (seq_len, batch_size, num_directions* hidden_size)
        ts_output = ts_output.permute(1,0,2) #[b,t, 256]
        ts_output = ts_output.reshape(-1, ts_output.shape[-1])#[b*t, 256]

        if self.text_model.name == 'cnn':
            text_output = self.text_model(text)
            text_output = self.timespread(text_output, ts_output, weight_mat).float()
            #print('rnn mean', torch.mean(rnn_output))
            text_output = text_output.reshape(-1, text_output.shape[-1]) #[b*t, 768]
        elif self.text_model.name == 'avg':
            text_output = text.reshape(-1, text.shape[-1])
        #print(text_output.is_cuda)
            
        #print('cnn mean', torch.mean(cnn_output))
        
        if not self.tabs:
            out = torch.cat((ts_output, text_output), dim =1)
        else:
            out = torch.cat((ts_output, text_output, tabs), dim =1)
        
       
        out_copy = out.reshape(batch_size, -1, out.shape[1]) # [b, t, 1024]
        ihm_out = out_copy[:, 48, :] #[b, 1024]
        
        decom_out = self.fc_decom(out) #[b*t, 2]
        los_out = self.fc_los(out) # [b*t, 10]
        ihm_out = self.fc_ihm(ihm_out)
        
        return decom_out, los_out, ihm_out


# class MultiModal_Model_4_task(nn.Module):
#     def __init__(self, cnn_model, lstm_model,tabs =False, text_postprocess = 'weighted_combine', ihm_classes=2, decom_classes=2, los_classes=10, pheno_classes = 25):
#         super(MultiModal_Model_4_task, self).__init__(cnn_model = cnn_model, lstm_model = lstm_model)
#         self.pheno_classes = pheno_classes
#         self.tabs = tabs
#         if not self.tabs:
#             self.fc_pheno = nn.Linear(1024, self.pheno_classes)
#         else:
#             self.tabs_net = nn.Sequential(nn.Linear(13, 5), nn.ReLU())
#             self.fc_pheno = nn.Linear(1029, self.pheno_classes)

#     def forward(self, text, ts, weight_mat =None, tabs = None):
#         batch_size = ts.shape[1]
#         t = ts.shape[0]
#         if self.tabs:
#             tabs = tabs.repeat(1, t, 1)
#             tabs = self.tabs_net(tabs).float()
#             tabs = tabs.reshape(-1, tabs.shape[-1])

        
#         cnn_output = self.cnn_model(text)
#         rnn_output = self.lstm_model(ts).float() # shape (seq_len, batch_size, num_directions* hidden_size)
#         cnn_output = self.timespread(cnn_output, rnn_output, weight_mat).float()
#         rnn_output = rnn_output.permute(1,0,2) #[b,t, 256]
#         rnn_output = rnn_output.reshape(-1, rnn_output.shape[-1])#[b*t, 256]
#         cnn_output = cnn_output.reshape(-1, cnn_output.shape[-1]) #[b*t, 768]
        
#         if not self.tabs:
#             out = torch.cat((rnn_output, cnn_output), dim =1)
#         else:
#             out = torch.cat((rnn_output, cnn_output, tabs), dim =1)
       
#         out_copy = out.reshape(batch_size, -1, out.shape[1]) # [b, t, 1024]
#         ihm_out = out_copy[:, 48, :] #[b, 1024]
#         pheno_out = out_copy[:, -1, :]
        
#         decom_out = self.fc_decom(out) #[b*t, 2]
#         los_out = self.fc_los(out) # [b*t, 10]
#         ihm_out = self.fc_ihm(ihm_out)
#         pheno_out = self.fc_pheno(pheno_out)
        
#         return decom_out, los_out, ihm_out, pheno_out
    


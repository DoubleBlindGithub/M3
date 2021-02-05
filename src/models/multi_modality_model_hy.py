import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F



class TaskSpecificComponent(nn.Module):
    """
    Base class for a task specific component
    Must be extended by user.
    """
    def __init__(self, name, loss):
        """
        name: str: the name of this task
        loss: nn.Functional: the loss function to use for this task
        """
        super(TaskSpecificComponent, self).__init__()
        self.name = name
        self.loss = loss

    def forward(self, x):
        raise NotImplementedError

    def get_loss(self, logits, labels):
        """
        Compute the loss for this Task
        """
        return self.loss(logits, labels)
class FCTaskComponent(TaskSpecificComponent):
    """
    Default task componenet. Fully connected layer with dropout
    """

    def __init__(self, name, loss, input_dim, output_dim, hidden_size=128, dropout=0.8):
        super(FCTaskComponent, self).__init__(name, loss)
        self.fc_model = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim)
        )
    def forward(self, x):
        return self.fc_model(x)

class ModalityEncoder(nn.Module):
    """
    Base class for a modality encoder
    Should not be instantiated on it's own
    embedding_dim is the size of the embedding  of this modaity
    """
    def __init__(self, name=None, output_dim=0):
        super(ModalityEncoder, self).__init__()
        self.embedding_dim = output_dim
        self.name = name
    
    def forward(self, x):
        raise NotImplementedError

class TabularEmbedding(ModalityEncoder):
    def __init__(self, inputs_dict, device):
        """
        inputs_dict: dict of ints. Evey key is a category of embeddings. Every value 
        is tuple of (the number of unique classes for that key, size for this embedding)
        """
        super(TabularEmbedding, self).__init__(name="Tabular")
        self.embd_dict = {}
        self.embedding_dim = 0#Embeddng lengh of all embeddings
        for k,v in inputs_dict.items():
            num_class, embd_size = v
            self.embd_dict[k] = nn.Embedding(num_class, embd_size, padding_idx = 0).to(device)
            self.embedding_dim += embd_size
    
    def forward(self, x_dict):
        embds = []
        for category in self.embd_dict:#x_dict may have more categories than we are using
            cat_input = x_dict[category]
            embds.append(self.embd_dict[category](cat_input))
        return torch.cat(embds, 1)


class LSTMModel(ModalityEncoder):
    def __init__(self, input_dim, hidden_dim, layers,dropout=0.0, bidirectional=False):
        super(LSTMModel, self).__init__(name="Time Series")
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout, bidirectional =bidirectional)
        self.embedding_dim = hidden_dim + bidirectional*hidden_dim
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x) # [seq_len, batch, hidden size]
        return out

class TextCNN(ModalityEncoder):
    def __init__(self, in_channels, out_channels, kernel_heights, embedding_length, name ='cnn'):
        super(TextCNN, self).__init__(name="cnn")
        
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
        self.Dropout = nn.Dropout(0.1)
        self.embedding_dim = out_channels * 3
        
        
    def conv_block(self, x, conv_layer):
        conv_out = conv_layer(x)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim)
        max_out = F.max_pool1d(activation, activation.shape[2]).squeeze(2)# maxpool_out.size() = (batch_size, out_channels)
    

        return max_out
    
    def forward(self, x):
        # x shape [batch_size, num_doc, sen_len, emb_dim]
        res = []
        #print(x.shape[1])
        for i in range(x.shape[1]):
            # i is the number of docs in this batch 
            # input.size() = (batch_size, num_seq, embedding_length)
            x_i = x[:,i,:,:].unsqueeze(1) #(batch_size, 1, len_seq, embedding_length)
            max_out1 = self.conv_block(x_i, self.conv1) # [batch_size, out_channel]
            #print('max_out shape', max_out1.shape)
            max_out2 = self.conv_block(x_i, self.conv2)
            #print('max_out 2 shape', max_out2.shape)
            max_out3 = self.conv_block(x_i, self.conv3)
            text_features = torch.cat((max_out1, max_out2, max_out3), 1) # [batch_size, out_channel*3]
            #text_features = max_out1+max_out2+max_out3
            #print(text_features)
            text_features = self.Dropout(text_features)
            res.append(text_features)  
        res = torch.stack(res, dim =1) #[batch_size, num_doc, 768]
        #print(res.shape)
        return res




class MultiModalEncoder(nn.Module):
    """
    The Multimodal encoder
    Has an encoder per each modality(time seriest(ts), text, and tabular(tab))
    Each encoder should inherit from ModalityEncoder or be None if that 
    modality isn't used
    """
    def __init__(self, ts_model, text_model, tab_model):
        super(MultiModalEncoder, self).__init__()
        self.ts_model = ts_model
        self.text_model = text_model
        self.tab_model = tab_model
        self.global_dim = 0
        if ts_model is not None:
            self.global_dim += self.ts_model.embedding_dim
        if text_model is not None:
            self.global_dim += self.text_model.embedding_dim
        if tab_model is not None:
            self.global_dim += self.tab_model.embedding_dim
    
    def combine(self, embeddings):
        """
        Combine all the modal embeddings into one tensor. By default just a concatentation
        embeddings: list: list of all embeddings, [ts_embd, text_embd, tab_embd]. Elements of list
                          are null if they are not used(no model is provided to encode that modality)
        """
        embeddings = list(filter(lambda x: x is not None, embeddings))
        assert len(embeddings) > 0, "No modality embeddings available"
        return torch.cat(embeddings, dim=1)
    
    def forward(self, ts =None, texts =None, texts_weight_mat =None, tab_dict = None):
        ts_output = None
        texts_output = None
        tab_output = None
        t = None

        if ts is not None:
            batch_size = ts.shape[1]
            t = ts.shape[0]
            ts[min(24, t):,:,:] *= 0

        
        if texts_weight_mat is not None:
            batch_size = texts_weight_mat.shape[0]
            t = texts_weight_mat.shape[1]
        
        if self.ts_model is not None:
            ts_output = self.ts_model(ts).float() # shape (seq_len, batch_size, num_directions* hidden_size)
            ts_output = ts_output.permute(1,0,2) #[b,t, 256]
            ts_output = ts_output.reshape(-1, ts_output.shape[-1])#[b*t, 256]

            # ts_norm_layer = nn.LayerNorm(ts_output.size()[1:], elementwise_affine=False)
            
        
        if self.text_model is not None: #self.use_text:
            if self.text_model.name == 'avg':
                texts_output = texts.reshape(-1, texts.shape[-1])
            else:
                assert texts_weight_mat is not None
                texts_output = self.text_model(texts)
                texts_output = self.batch_weighted_combination(texts_output, t, texts_weight_mat)
                texts_output = texts_output.reshape(-1, texts_output.shape[-1]) #[b*t, 768]

            text_norm_layer = nn.LayerNorm(texts_output.size()[1:], elementwise_affine=False)
            texts_output = text_norm_layer(texts_output)
            
        
        out = None
        if self.ts_model is not None: #self.use_ts:
            out = ts_output
        
        #if True: #self.use_text:
        #    if out is None:
        #        out = texts_output
        #    else:
        #        out = torch.cat((out, texts_output), dim=1)
        
        if self.tab_model is not None:#self.use_tab:
            tab_out = self.tab_model(tab_dict)
            tab_out = torch.unsqueeze(tab_out, dim=1)
            #t = out.shape[0]//tab_out.shape[0] # b*t/b
            if t is None:
                t = 1
            expanded_tab_out = tab_out.expand(-1, t, self.tab_model.embedding_dim)
            tab_output = expanded_tab_out.reshape(-1, self.tab_model.embedding_dim)
            #out = torch.cat((out, tab_out.reshape(-1, self.tab_model.embedding_dim)), dim=1)
        return self.combine([ts_output, texts_output, tab_output])

    def batch_weighted_combination(self, features, time, weight_mat = None):
        res = []
        for i in range(features.shape[0]):
            feat_i = features[i,:,:]
            weight_i = weight_mat[i,:,:]
            new_feat_i = FeatureSpreadWTime(feat_i, weight_i)
            res.append(new_feat_i)
        res = torch.stack(res, dim=0)
        return res
 
    
class MultiModalMultiTaskWrapper(nn.Module):
    """
    The Entire MM-MT Model
    """
    def __init__(self, mm_encoder, ihm_model, decomp_model,
                 los_model, pheno_model, readmit_model, ltm_model):
        super(MultiModalMultiTaskWrapper,self).__init__()
        #--------Modalities-------------#
        self.mm_encoder = mm_encoder

        #--------Tasks-----------------#
        self.decomp_model = decomp_model
        self.los_model = los_model
        self.ihm_model = ihm_model
        self.pheno_model = pheno_model
        self.readmit_model = readmit_model
        self.ltm_model = ltm_model


    def batch_weighted_combination(self, features, time, weight_mat = None):
        res = []
        for i in range(features.shape[0]):
            feat_i = features[i,:,:]
            weight_i = weight_mat[i,:,:]
            new_feat_i = FeatureSpreadWTime(feat_i, weight_i)
            res.append(new_feat_i)
        res = torch.stack(res, dim=0)
        return res
        

    def forward(self, ts =None, texts =None, texts_weight_mat =None, tab_dict = None):
        if ts is not None:
            batch_size = ts.shape[1]
            t = ts.shape[0]
        
        if texts_weight_mat is not None:
            batch_size = texts_weight_mat.shape[0]
            t = texts_weight_mat.shape[1]
 
        out = self.mm_encoder(ts, texts, texts_weight_mat, tab_dict)

        out_copy = out.reshape(batch_size, -1, out.shape[1]) # [b, t, 1124]

        #----------- Select the appropiate timepoint per task ------------#
        ihm_out = out_copy[:, 47, :] #[b, 1124]
        los_out = out_copy[:, 23, :] #[b, 1124]
        pheno_out = out_copy[:, -1, :]
        readmit_out = out_copy[:, -1, :]
        ltm_out = out_copy[:,-1,:]

        #---------- Compute logits per every task ----------------------------#
        decomp_out = self.decomp_model(out) #[b*t, 2]
        los_out = self.los_model(los_out) # [b*t, 10]
        ihm_out = self.ihm_model(ihm_out)
        pheno_out = self.pheno_model(pheno_out)
        readmit_out = self.readmit_model(readmit_out)
        ltm_out = self.ltm_model(ltm_out)

        
        return decomp_out, los_out, ihm_out, pheno_out, readmit_out, ltm_out





class AttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers,dropout=0.0, bidirectional=False):
        super(AttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout, bidirectional =bidirectional)

    def attention_net(self, lstm_output, final_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        
        Arguments
        ---------
        
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        
        ---------
        
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.
                  
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                      
        """
        
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state

    def forward(self, x):
        output, (final_hidden_state, final_cell_state) = self.lstm(x)
        attn_output = self.attention_net(output, final_hidden_state)
        return attn_output


class ChannelWiseLSTM(nn.Module):
    def __init__(self, preprocess_dim, hidden_dim, layers, header, bidirectional = False):
        super(ChannelWiseLSTM, self).__init__()
        self.preprocess_dim = preprocess_dim
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.bidirectional = bidirectional

        # Parse channels
        channel_names = set()
        for ch in header:
            if ch.find("mask->") != -1:
                continue
            pos = ch.find("->")
            if pos != -1:
                channel_names.add(ch[:pos])
            else:
                channel_names.add(ch)
        channel_names = sorted(list(channel_names))
        print("==> found {} channels: {}".format(len(channel_names), channel_names))

        channels = []  # each channel is a list of columns
        for ch in channel_names:
            indices = range(len(header))
            indices = list(filter(lambda i: header[i].find(ch) != -1, indices))
            channels.append(indices)

        self.postprocess_dim = preprocess_dim*len(channels)
        if bidirectional:
            self.postprocess_dim = self.postprocess_dim*2
        self.channels = channels
        self.embedding_dim = hidden_dim

        self.preprocess_lstm = nn.ModuleList()
        for i in range(len(self.channels)):
            input_dim = len(self.channels[i])
            out_dim = self.preprocess_dim
            self.preprocess_lstm.append(nn.LSTM(input_dim, out_dim, layers, bidirectional =bidirectional))

        self.postprocess_lstm = nn.LSTM(self.postprocess_dim, hidden_dim, layers, bidirectional =False)

    
    def forward(self, x):
        # x shape (t, batch, 76)
        pre_process_res = []
        for i in range(len(self.channels)):
            idx = torch.tensor(self.channels[i]).cuda()
            x_i = torch.index_select(x, 2, idx)
            x_i,_ = self.preprocess_lstm[i](x_i)
           # print(x_i.shape)
            pre_process_res.append(x_i)
        pre_process_res = torch.cat(pre_process_res, dim=2)
        post_process_res, _ = self.postprocess_lstm(pre_process_res)
        return post_process_res


class GRU_D(nn.Module):
    pass

class Text_AVG(nn.Module):
    def __init__(self, name = 'avg'):
        """
        The actual average operation is done before feed it to network due to memory considerations
        """
        super(Text_AVG, self).__init__()
        self.name = name
        self.embedding_dim = 200

    def forward(self,x):
        return x


class Text_RNN(nn.Module):
    def __init__(self, embedding_length, hidden_size, name = 'rnn'):
        super(Text_RNN, self).__init__()
        self.name = name
        self.embedding_length = embedding_length
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(embedding_length, hidden_size, num_layers=2, bidirectional=False)
        
    def forward(self, x):
        #batch_size = x.shape[1]
        res = []
        #h_0 =  Variable(torch.zeros(4, batch_size, self.hidden_size).cuda()),
        #print(x.shape[1])
        for i in range(x.shape[1]):
            x_i = x[:,i,:,:]
            #print(x_i.shape)
            x_i = x_i.permute(1, 0, 2)
            output, h_n = self.rnn(x_i) 
            # h_n.size() = (2, batch_size, hidden_size), 2 = num_layers*num_directions
            h_n = h_n.permute(1, 0, 2)
            h_n = h_n.contiguous().view(h_n.size()[0], -1)
            # h_n.size() = (batch_size, 2*hidden_size)
            res.append(h_n)
        res = torch.stack(res, dim =1)
        return res


class LSTMAttentionModel(torch.nn.Module):
    def __init__(self, hidden_size, embedding_length, name = 'lstm attn'):
        super(LSTMAttentionModel, self).__init__()
        
        """
        Arguments
        ---------
        hidden_sie : Size of the hidden_state of the LSTM
        embedding_length : Embeddding dimension of GloVe word embeddings
        --------
        
        """
        
        self.hidden_size = hidden_size
        self.embedding_length = embedding_length
        self.lstm = nn.LSTM(embedding_length, hidden_size)
        self.name = name
        #self.attn_fc_layer = nn.Linear()
        
    def attention_net(self, lstm_output, final_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        
        Arguments
        ---------
        
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        
        ---------
        
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.
                  
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                      
        """
        
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state
    
    def forward(self, x):
    
        """ 
        Parameters
        ----------
        x: input_sentence of shape = (batch_size, num_doc, len_sequences, emb_dim)
        Returns
        -------
        Output of the linear layer containing logits for pos & neg class which receives its input as the new_hidden_state which is basically the output of the Attention network.
        final_output.shape = (batch_size, output_size)
        
        """
        res = []       
        for i in range(x.shape[1]):
            x_i = x[:, i, :, :]
            x_i = x_i.permute(1, 0, 2)
            output, (final_hidden_state, final_cell_state) = self.lstm(x_i)
            output = output.permute(1, 0, 2)
            attn_output = self.attention_net(output, final_hidden_state)
            res.append(attn_output)
        res = torch.stack(res, dim =1)
        return res
        # res = []
        # for i in range(x.shape[1]):
        #     x_i = x[:,i,:,:]
        #     x_i = x_i.permute(1, 0, 2)
        #     output, (final_hidden_state, final_cell_state) = self.lstm(x_i) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        #     output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        #     attn_output = self.attention_net(output, final_hidden_state)
        #     res.append(attn_output)
        # res = torch.stack(res, dim =1)
        # return res





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
    A = features_i.permute(1,0) #[200, n]
    #norm = (weight_i != 0).sum(dim=1).reshape(-1,1)+1
    #print(norm.shape)
    
    B = weight_i.unsqueeze(1) # [t, 1, n]
    C = A*B
    C = torch.sum(C, dim=2) # t*200
    # here it should be normalized by the num of non-zero elements, but in my experiments, the following normalization works better
    C/= A.shape[1]
    #C/= norm
    return C

def batched_featurespredwtime(sentence_feature, weight):
    A = sentence_feature.permute(0,2,1)
    A = A.unsqueeze(1)
    B = weight.unsqueeze(2)
    C = A*B
    C = torch.sum(C, dim = 3)
    C/= A.shape[2]
    return C

def mean_text(sentence_features, dim):
    return torch.mean(sentence_features, dim = dim, keepdim = True)





class MultiModal_Multitask_Model(nn.Module):
    def __init__(self, ts_model, text_model, tab_model, use_ts, use_text, use_tab, regression = False, \
        ihm_classes=2,decomp_classes=2, los_classes=3, pheno_classes = 25, readmit_classes = 5, ltm_classes = 2):
        super(MultiModal_Multitask_Model, self).__init__()
        self.ts_model = ts_model
        self.text_model = text_model
        self.tab_model = tab_model
        self.decomp_classes = decomp_classes
        self.readmit_classes = readmit_classes
        self.los_classes = los_classes
        self.ltm_classes = ltm_classes
        if regression:
            self.los_classes = 1
        self.ihm_classes = ihm_classes
        self.pheno_classes = pheno_classes
        self.use_ts = use_ts
        self.use_text = use_text
        self.use_tab = use_tab
        
        if ts_model.bidirectional:
            tsout_dim = self.ts_model.hidden_dim*2
        else:
            tsout_dim = self.ts_model.hidden_dim
        
        if self.text_model.name == 'cnn':
            textout_dim = self.text_model.out_channels*3
        elif self.text_model.name == 'avg':
            textout_dim = 200
        else:
            textout_dim = self.text_model.hidden_size*2
        
        

        out_dim = 0
        if self.use_ts:
            out_dim+=tsout_dim
        if self.use_text:
            out_dim+=textout_dim
        
        if self.use_tab:
            out_dim+= self.tab_model.embedding_dim
      
        self.fc_decomp = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(128, self.decomp_classes))

        self.fc_los = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(32, self.los_classes))


        self.fc_ihm = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(64, self.ihm_classes))
        
        self.fc_pheno = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(out_dim, self.pheno_classes))
            
            # nn.ReLU(),
            # nn.Linear(512, self.pheno_classes))

        self.fc_readmit = nn.Sequential(
          
            nn.Dropout(0.3),
            
            nn.Linear(out_dim, self.readmit_classes))



    def batch_weighted_combination(self, features, time, weight_mat = None):
        res = []
        for i in range(features.shape[0]):
            feat_i = features[i,:,:]
            weight_i = weight_mat[i,:,:]
            new_feat_i = FeatureSpreadWTime(feat_i, weight_i)
            res.append(new_feat_i)
        res = torch.stack(res, dim=0)
        return res
        

    def forward(self, ts =None, texts =None, texts_weight_mat =None, tab_dict = None):
        if ts is not None:
            batch_size = ts.shape[1]
            t = ts.shape[0]
        
        if texts_weight_mat is not None:
            batch_size = texts_weight_mat.shape[0]
            t = texts_weight_mat.shape[1]
        
        if self.use_ts:
            ts_output = self.ts_model(ts).float() # shape (seq_len, batch_size, num_directions* hidden_size)
            ts_output = ts_output.permute(1,0,2) #[b,t, 256]
            ts_output = ts_output.reshape(-1, ts_output.shape[-1])#[b*t, 256]

            # ts_norm_layer = nn.LayerNorm(ts_output.size()[1:], elementwise_affine=False)
            
        
        if self.use_text:
            if self.text_model.name == 'avg':
                texts_output = texts.reshape(-1, texts.shape[-1])
            else:
                assert texts_weight_mat is not None
                texts_output = self.text_model(texts)
                texts_output = self.batch_weighted_combination(texts_output, t, texts_weight_mat)
                texts_output = texts_output.reshape(-1, texts_output.shape[-1]) #[b*t, 768]

            text_norm_layer = nn.LayerNorm(texts_output.size()[1:], elementwise_affine=False)
            texts_output = text_norm_layer(texts_output)
            
        
        out = None
        if self.use_ts:
            out = ts_output
        
        if self.use_text:
            if out is None:
                out = texts_output
            else:
                out = torch.cat((out, texts_output), dim=1)
        
        if self.use_tab:
            tab_out = self.tab_model(tab_dict)
            tab_out = torch.unsqueeze(tab_out, dim=1)
            t = out.shape[0]//tab_out.shape[0] # b*t/b
            append = tab_out.expand(-1, t, self.tab_model.embedding_dim)
            out = torch.cat((out, append.reshape(-1, self.tab_model.embedding_dim)), dim=1)


        out_copy = out.reshape(batch_size, -1, out.shape[1]) # [b, t, 1124]

        ihm_out = out_copy[:, 47, :] #[b, 1124]
        los_out = out_copy[:, 23, :] #[b, 1124]
        pheno_out = out_copy[:, -1, :]
        readmit_out = out_copy[:, -1, :]
        ltm_out = out_copy[:,-1,:]
        
        decomp_out = self.fc_decomp(out) #[b*t, 2]
        los_out = self.fc_los(los_out) # [b*t, 10]
        ihm_out = self.fc_ihm(ihm_out)
        pheno_out = self.fc_pheno(pheno_out)
        readmit_out = self.fc_readmit(readmit_out)
        ltm_out = self.fc_ltm(ltm_out)

        
        return decomp_out, los_out, ihm_out, pheno_out, readmit_out, ltm_out


class Text_Only_DS(nn.Module):
    def __init__(self, text_model, ihm_classes=2,decomp_classes=2, los_classes=3, pheno_classes = 25, readmit_classes = 2):
        super(Text_Only_DS, self).__init__()
        
        self.text_model = text_model
        self.decomp_classes = decomp_classes
        self.readmit_classes = readmit_classes
        self.los_classes = los_classes
        
        self.ihm_classes = ihm_classes
        self.pheno_classes = pheno_classes
       
       
        if self.text_model.name == 'cnn':
            textout_dim = self.text_model.out_channels*3
        elif self.text_model.name == 'avg':
            textout_dim = 200
        else:
            textout_dim = self.text_model.hidden_size*2
        

        out_dim = textout_dim
      
        self.fc_decomp = nn.Sequential(
            nn.Linear(out_dim, 128),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(128, self.decomp_classes))

        self.fc_los = nn.Sequential(
            nn.Linear(out_dim, 32),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(32, self.los_classes))


        self.fc_ihm = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(64, self.ihm_classes))
        
        self.fc_pheno = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(out_dim, self.pheno_classes))
            
            # nn.ReLU(),
            # nn.Linear(512, self.pheno_classes))

        self.fc_readmit = nn.Sequential(
          
            nn.Dropout(0.3),
            
            nn.Linear(out_dim, self.readmit_classes))

    

    def forward(self, texts):
        #print('text shape', texts.shape)
       
        
 
        if self.text_model.name == 'avg':
            texts_output = texts.reshape(-1, texts.shape[-1])
        else:
           
            texts_output = self.text_model(texts)
            texts_output = texts_output.reshape(-1, texts_output.shape[-1]) #[b, 768]

        


        readmit_out = self.fc_readmit(texts_output)

        
        return  readmit_out

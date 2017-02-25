-------------------------------
require('nn')
require('rnn')
dofile('model/struct_layer.lua')
require('model/seq_blstmp.lua')
require('model/seq_bgru.lua')
-------------------------------

-------------------------------
local model = {}

-- build model
function model:build_model(arc_type, model_type, input_dim, hidden_size, dropout_rate, num_layers, pretrained_path)
  local m = nn.Sequential()
  if model_type == 'gru' then
    obj = nn.SeqBGRU
    if arc_type == 'rnn' then
      obj = nn.SeqGRU
    end  
  elseif model_type == 'lstmp' then
    obj = nn.SeqBLSTMP
    if arc_type == 'rnn' then
      obj = nn.SeqLSTMP
    end  
  else
    obj = nn.SeqBRNN
    if arc_type == 'rnn' then
      obj = nn.SeqLSTM
    end  
  end
  
  -- load pretrained model if requested
  if pretrained_path then
    m = torch.load(pretrained_path)
  
  -- build the model by the given architecture 
  else
    if arc_type == 'rnn' then
      if model_type == 'lstmp' then
        m:add(obj(input_dim, hidden_size, hidden_size))
      else
        m:add(obj(input_dim, hidden_size))
      end
      m:add(nn.Sequencer(nn.NormStabilizer()))
      m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
      for i=2, num_layers do
        m:add(obj(hidden_size, hidden_size))
        m:add(nn.Sequencer(nn.NormStabilizer()))
        m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
      end
      m:add(nn.StructLayer(hidden_size, method))
      
    else 
      if arc_type ~= 'brnn' then
        print('unkown model, set model to BRNN.')
      end
      -- define merge function
      local merge = nn.JoinTable(2, 2)
      --local merge = nn.CSubTable()      
      -- add the desire amount of layers + dropout + norm stabilizer as regularization
      m:add(obj(input_dim, hidden_size, false, merge))  
      m:add(nn.Sequencer(nn.NormStabilizer()))
      m:add(nn.Sequencer(nn.Dropout(dropout_rate)))  
      for i=2, num_layers do
        m:add(obj(2*hidden_size, hidden_size, false, merge))
        m:add(nn.Sequencer(nn.NormStabilizer()))
        m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
      end  
      m:add(nn.StructLayer(2*hidden_size))      
    end
  end
  
  return m

  --[[
  m = nn.Sequential()
  m:add(nn.SeqLSTM(input_dim, hidden_size))
  m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
  for i=2, num_layers do
    m:add(nn.SeqLSTM(hidden_size, hidden_size))
    m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
  end
  m:add(nn.StructLayer(hidden_size, method))
  ]]--
  -- m = torch.load('results/vot/bb_pos/model_14/n_model.net')
  --return m

  --[[  
  m_o = torch.load('results/vot/pre_train/model.net')
  
  local p, gp = m_o.modules[1]:getParameters()
  local np, np_g = m.modules[1]:getParameters()    
  np = p:clone() 
  
  p, gp = m_o.modules[3]:getParameters()
  np, np_g = m.modules[3]:getParameters()  
  np = p:clone()
  ]]--    
end

return model
-------------------------------
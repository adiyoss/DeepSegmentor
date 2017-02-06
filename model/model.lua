-------------------------------
require('nn')
require('rnn')
dofile('model/struct_layer.lua')
-------------------------------

-------------------------------
local model = {}

-- build model
function model:build_model(input_dim, hidden_size, dropout_rate, method, num_layers)
  
  --[[
  -- adds the output of one time-step of fwd and bwd rnns.    
  m = nn.Sequential()
  --local merge = nn.CSubTable()
  local merge = nn.JoinTable(2, 2)
  -- add the desire amount of layers
  m:add(nn.SeqBRNN(input_dim, hidden_size, false, merge))  
  --m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
  --m:add(nn.Sequencer(nn.NormStabilizer()))
  for i=2, num_layers do
    m:add(nn.SeqBRNN(2*hidden_size, hidden_size, false, merge))
    --m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
    --m:add(nn.Sequencer(nn.NormStabilizer()))
  end  
  m:add(nn.StructLayer(2*hidden_size, method))  
  return m
  ]]--
  m = nn.Sequential()
  m:add(nn.SeqLSTM(input_dim, hidden_size))
  m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
  for i=2, num_layers do
    m:add(nn.SeqLSTM(hidden_size, hidden_size))
    m:add(nn.Sequencer(nn.Dropout(dropout_rate)))
  end
  m:add(nn.StructLayer(hidden_size, method))
  
  -- m = torch.load('results/vot/bb_pos/model_14/n_model.net')
  return m

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
-------------------------------
require('nn')
require('rnn')
dofile('model/struct_layer.lua')
-------------------------------

-------------------------------
local model = {}

-- build model
function model:build_model(input_dim, hidden_size, dropout_rate, method, num_layers)
  -- adds the output of one time-step of fwd and bwd rnns.    
  m = nn.Sequential()  
  local merge = nn.CAddTable()  
  
  -- add the desire amount of layers
  m:add(nn.SeqBRNN(input_dim, hidden_size, false, merge)):add(nn.Sequencer(nn.Dropout(dropout_rate)))
  for i=2, num_layers do
    m:add(nn.SeqBRNN(hidden_size, hidden_size, false, merge)):add(nn.Sequencer(nn.Dropout(dropout_rate)))
  end  
  m:add(nn.StructLayer(hidden_size, method))
  
  return m
end

return model
-------------------------------
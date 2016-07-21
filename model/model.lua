-------------------------------
require('nn')
require('rnn')
dofile('model/struct_layer.lua')
-------------------------------

-------------------------------
local model = {}

-- build model
function model:build_model(input_dim, hidden_size, dropout_rate, method)
  -- adds the output of one time-step of fwd and bwd rnns.    
  local merge = nn.CAddTable()
  brnn = nn.SeqBRNN(input_dim, hidden_size, false, merge)
  
  m = nn.Sequential()
     :add(brnn)
     :add(nn.Sequencer(nn.Dropout(dropout_rate)))          
     :add(nn.StructLayer(hidden_size, method))     
  return m
end

return model
-------------------------------
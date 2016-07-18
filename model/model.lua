-------------------------------
require('nn')
require('rnn')
dofile('model/struct_layer.lua')
-------------------------------

-------------------------------
local model = {}
  
-- build model
function model:build_model(input_dim, hidden_size, dropout_rate, method)
  
  --[[
  -- forward rnn
  -- build LSTM based rnn
  local fwd = nn.FastLSTM(input_dim, hidden_size)
  
  -- backward rnn (will be applied in reverse order of input sequence)
  local bwd = fwd:clone()
  
  bwd:reset() -- reinitializes parameters
    -- build the bidirectional lstm
  local brnn = nn.BiSequencer(fwd, bwd, merge)
  ]]--
  
  -- merges the output of one time-step of fwd and bwd rnns.  
  -- local merge = nn.CSubTable()
  local merge = nn.JoinTable(1, 1)  
  brnn = nn.SeqBRNN(input_dim, hidden_size, false, merge)
  
  m = nn.Sequential()
     :add(brnn)
     --:add(nn.Sequencer(nn.NormStabilizer()))
     :add(nn.Sequencer(nn.Dropout(dropout_rate)))          
     :add(nn.StructLayer(2*hidden_size, method))
     
  return m
end

return model
-------------------------------
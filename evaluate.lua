require 'torch'
require 'xlua'

local evaluator = {}

function evaluator:evaluate(model, criterion, x, y, f_n, verbose)
  local avg_score = 0
  local cumulative_loss = 0
  local durations = {}
  
  model:evaluate()
  for t =1, #x do
      if not verbose then
        xlua.progress(t, #x)
      end
      if t == 3 then
        print(1)
      end
      local output = model:forward(x[t])
      local score, onsets = criterion:predict(output, #y[t])
      local loss = criterion:task_loss(y[t], onsets)
      --durations[t] = torch.abs((tonumber(offset) - tonumber(onset)) - (tonumber(y[t][2]) - tonumber(y[t][1])))
      avg_score = avg_score + score
      cumulative_loss = cumulative_loss + loss      
      -- printings
      if verbose then
        print('Filename: ' .. f_n[t] .. ', score: ' .. score .. ', loss: ' .. loss)
      end
  end
  cumulative_loss = cumulative_loss / #x
  avg_score  = avg_score / #x
  -- printings
  if verbose then
    print('\nAverage score: ' .. avg_score)
    print('Average cumulative loss: ' .. cumulative_loss)
  end  
  return cumulative_loss, avg_score, durations
end

function evaluator:plot_vot_stats(durations)  
  ms_2 = 0
  ms_5 = 0
  ms_10 = 0
  ms_15 = 0
  ms_25 = 0
  ms_50 = 0

  for d=1, #durations do
    if durations[d] <= 2 then
      ms_2 = ms_2 + 1
    end
    if durations[d] <= 5 then
      ms_5 = ms_5 + 1  
    end
    if durations[d] <= 10 then
      ms_10 = ms_10 + 1  
    end
    if durations[d] <= 15 then
      ms_15 = ms_15 + 1  
    end
    if durations[d] <= 25 then
      ms_25 = ms_25 + 1  
    end
    if durations[d] <= 50 then
      ms_50 = ms_50 + 1  
    end
  end
  print('|(y_off - y_on) - (y_hat_off - y_hat_on)|')
  print('Percentage of <= 2ms: ' .. (ms_2 / #durations) .. '%')
  print('Percentage of <= 5ms: ' .. (ms_5 / #durations) .. '%')
  print('Percentage of <= 10ms: ' .. (ms_10 / #durations) .. '%')
  print('Percentage of <= 15ms: ' .. (ms_15 / #durations) .. '%')
  print('Percentage of <= 25ms: ' .. (ms_25 / #durations) .. '%')
  print('Percentage of <= 50ms: ' .. (ms_50 / #durations) .. '%')
end

return evaluator
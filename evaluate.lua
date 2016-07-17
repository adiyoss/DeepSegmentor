local evaluator = {}

function evaluator:evaluate(model, criterion, x, y, f_n, verbose)  
  local avg_score = 0
  local cumulative_loss = 0
  
  model:evaluate()  
  for t =1, #x do
      local output = model:forward(x[t])
      local score, onset, offset = criterion:predict(output)
      local loss = criterion:task_loss(y[t], {onset, offset})
      avg_score = avg_score + score
      cumulative_loss = cumulative_loss + loss      
      -- printings
      if verbose then
        print('Filename: ' .. f_n[t] .. ', score: ' .. score .. ', y_hat: [' .. onset .. ', ' .. offset ..'], y: [' .. y[t][1] .. ', ' .. y[t][2] .. '], loss: ' .. loss)
      end
  end
  cumulative_loss = cumulative_loss / #x
  avg_score  = avg_score / #x
  -- printings
  if verbose then
    print('\nAverage score: ' .. avg_score)
    print('Average cumulative loss: ' .. cumulative_loss)
  end  
  return cumulative_loss, avg_score
end

return evaluator
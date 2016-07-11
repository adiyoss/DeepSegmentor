-------------------------------
--require('mobdebug').start()
require ('torch')
require ('nn')
utils = dofile('common/utils.lua')
-------------------------------

-------------------------------
local data = {}

-- C'Tor
function data:new()  
  self.x_suffix = '.data'
  self.y_suffix = '.labels'
end

-- read data
function data:read_data_synthetic(path_x, path_y, input_dim)
  local x = {}
  local y = {}
  
  for file in paths.files(path_x) do
    if string.sub(file,-string.len(self.x_suffix))==self.x_suffix then
      local x_filename = paths.concat(path_x, file)
      local y_filename = paths.concat(path_y, (string.sub(file, 0, string.len(file)-string.len(self.x_suffix)) .. self.y_suffix))
      local x_t = utils:load_data(x_filename, input_dim)
      local y_t = utils:load_labels(y_filename)
      table.insert(x, x_t)
      table.insert(y, y_t)
    end
  end
  
  return x, y
end

-- read data
function data:read_data_word_duration(path_x, path_y, input_dim)
  local x = {}
  local y = {}
    
  local x_all = torch.load(path_x)
  local y_all = torch.load(path_y)  
  
  x_all = x_all[{{}, {1, input_dim}}]
  
  x_all = x_all:reshape(3551, 256, input_dim)
  y_all = y_all:reshape(3551, 256)
  
  c = 0  
  for i=1,3551 do
    local flag = 0
    local onsets = {}
    for j=2,256 do
      if y_all[i][j-1] ~= y_all[i][j] then
        flag = flag + 1
        table.insert(onsets, j)
      end
    end
    if flag == 2 then
      local row = {}
      for j=1,x_all[i]:size(1) do
        table.insert(row, x_all[i][j]:reshape(1, input_dim))
      end      
      table.insert(x, row)
      table.insert(y, onsets)
      c = c + 1
    end
  end
    
  return x, y
end

return data
-------------------------------
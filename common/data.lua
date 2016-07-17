-------------------------------
require ('torch')
require ('nn')
utils = dofile('common/utils.lua')
logger = dofile('common/logger.lua')
-------------------------------

-------------------------------
local data = {}

-- C'Tor
function data:new()
  self.x_suffix = '.txt'
  self.y_suffix = '.labels'
end

-- read data
function data:read_data(path_x, path_y, input_dim)
  -- validation
  if not self.x_suffix or not self.y_suffix then
    logger:error('object was not initialized. call new() function before using data module.\n') 
  end
  
  local x = {}
  local y = {}
  local f_n = {}
  for file in paths.files(path_x) do
    if string.sub(file,-string.len(self.x_suffix))==self.x_suffix then
      local x_filename = paths.concat(path_x, file) 
      local y_filename = paths.concat(path_y, (string.sub(file, 0, string.len(file)-string.len(self.x_suffix)) .. self.y_suffix))
      local x_t = utils:load_data(x_filename, input_dim)
      local y_t = utils:load_labels(y_filename)
      table.insert(x, x_t)
      table.insert(y, y_t)      
      table.insert(f_n, file)
    end
  end  
  return x, y, f_n
end

return data
-------------------------------

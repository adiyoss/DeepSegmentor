-------------------------------
local logger = {}

-- print to standard error
function logger:error(msg)
   print('')
   io.stderr:write(msg)
   print('')
   os.exit(1)
end


return logger
-------------------------------

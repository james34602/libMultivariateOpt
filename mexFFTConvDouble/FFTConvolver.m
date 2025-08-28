classdef FFTConvolver < handle
    properties (GetAccess=public, SetAccess=protected)
        rb
        freed
    end
    methods % public methods
        function obj = FFTConvolver()
            obj.rb = FFTConvolverInit();
            obj.freed = 0;
        end
        function delete(obj)
            if ~obj.freed
                FFTConvolverFree(obj.rb);
                obj.freed = 1;
            end
        end
        function load(obj, blockSize, ir)
            if ~obj.freed
                if ~isa(ir, 'double')
                    ir = double(ir);
                end
                FFTConvolverLoad(obj.rb, blockSize, ir);
            end
        end
        function update(obj, ir)
            if ~obj.freed
                if ~isa(ir, 'double')
                    ir = double(ir);
                end
                FFTConvolverLoadRefreshNewIR(obj.rb, ir);
            end
        end
        function y = process(obj, data)
            if ~obj.freed
                if ~isa(data, 'double')
                    data = double(data);
                end
                y = FFTConvolverProcess(obj.rb, data);
            end
        end
        function processNoReturn(obj, data)
            if ~obj.freed
                if ~isa(data, 'double')
                    data = double(data);
                end
                FFTConvolverProcessNoReturn(obj.rb, data);
            end
        end
    end
end
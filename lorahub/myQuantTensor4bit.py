
import bitsandbytes as bnb
class myQuantTensor4bit:
    def __init__(self, tensor):
        self.quantized_tensor = None
        self.quant_state=None
        
        self.quantize(tensor)
        self.dtype = self.quantized_tensor.dtype
    
    def quantize(self,tensor):
        quant_tensor,quant_state = bnb.functional.quantize_nf4(tensor)
        del tensor
        self.quantized_tensor = quant_tensor
        self.quant_state = quant_state
    
    def dequantize(self):
        return bnb.functional.dequantize_nf4(self.quantized_tensor,self.quant_state)
    
    #override print
    def __str__(self):
        return str(self.quantized_tensor)
    #override multiplication
    def __mul__(self, other):
        return self.dequantize() * other
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __add__(self, other):
        return self.dequantize() + other
    
    def __radd__(self, other):
        return self.__add__(other)
'''Base model class for all other model variants.'''
class BaseModel(object):
    def forward(self, input):
        # forward function for CNN models should take in a tensor batch of input data
        # and return a batch of raw logits.
        raise NotImplementedError

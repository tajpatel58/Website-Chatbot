# Torchserve:

Torchserve is a way of deploying/serving a PyTorch model. This is my first deployment so I'll be writing up some notes for future use. 

The approach I followed here was: 

- Train and store model as .PTH file. 
- Build a custom handler to deal with requests. 
- Create bash script which will archive our model into a .MAR file.
- Create another bash script to host the model. 

### Training and Storing Model:
- Store/save model using torch.save. 
- Can store any variables, hyperparams and model.state_dict() into a dictionary and dump this using torch.save. Doing this, allows us to get our model functioning with only one .pth file. 


### Handlers:
- Torchserve requires a handler class, which inherits from BaseHandler. 
- Handlers are used to deal with requests, in particular they handle the pre-processing, inference and the postprocessing. 
- Things to know for building a custom handler:
    1. Need to have certain functions: initialize, preprocess, inference, postprocess, handle. 
    2. Each of these required functions have some essentials. 
        - Initialize: Can create and initialize any model parameters, stemmers, tools etc. Must have a positional argument named: context. (This stores model properties.)
        - Preprocess: Positional argument called data. Which is what is recieved from a post request. Need to use .get() to get the datapoint. May need to decode to access datapoint, depends on data structure. 
        - Postprocess: Deals with vector after inference. 
        - Handle: Takes in args: data, context. Makes calls to preprocess, inference, postprocess with the datapoint to output a prediction. Note the prediction/output of this function has to be a list!!
    

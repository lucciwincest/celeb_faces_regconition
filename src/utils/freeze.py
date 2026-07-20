def freeze (module):
    for p in module.parameters ():
        p.requires_grad = False

def unfreeze (module):
    for p in module.parameters ():
        p.requires_grad = True
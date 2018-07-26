
# import sys
# sys.path.append('/home/stas/fast.ai')
# from myutils.data_check import *


# md = ColumnarModelData.from_data_frame(...)
# check_my_model_data(md):
def check_my_model_data(md):
    print("\n***My Model Data***\n")
 
    # XXX: type of variables (int, long, etc)
    # XXX: y

    ds_objs  = [md.trn_ds, md.val_ds]
    ds_names = ["train", "valid"]
    if md.test_dl: 
        ds_objs.append(md.test_ds)
        ds_names.append("test ")
        
    print("Continuous: shape")
    for ds,name in zip(ds_objs, ds_names): print(f"  {name}:  {ds.conts.shape}")
    print("Categoric: shape")
    for ds,name in zip(ds_objs, ds_names): print(f"  {name}:  {ds.cats.shape}")
        
    # check categorical variables to have the same # of categories in train, valid and test sets
    # in case apply_cats() wasn't run or something else went wrong
    # XXX: see Parch column issue (different ranges in test / train) - no apply_cats()
    # Also check what happens if new values appear in test and weren't trained on?
    #for ds,name in zip(ds_objs, ds_names): print(f"  {name}:  {len(ds.cats)}")


#m = md.get_learner(...)
#check_my_learner(m)
def check_my_learner(learner):
    m = learner.model
    print("\n***My Learner***\n")

    mode = "Regression" if m.is_reg else "Classification" 
    # XXX: In case of classification can we check that y_range matches out_sz?
    print(f"{mode} model with out_sz (n_classes) = {m.outp.out_features} / y_range = {m.y_range}")
 

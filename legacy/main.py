import argparse
import os
import yaml
import sys
import importlib
import cofi_core
import numpy as np
from cofi_core import Model, Parameter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="  CoFI main launcher.")
    parser.add_argument(
        "PATH_TO_YAML", type=str, help="Path to a yaml file specifying your problem"
    )

    args = parser.parse_args()
    yamlpath = args.PATH_TO_YAML
    if not os.path.exists(yamlpath):
        print(f"Specified yaml file {yamlpath} does not exist")
        sys.exit(1)
    elif not os.path.isfile(yamlpath):
        print(f"Speciified yaml path {yamlpath} is not a regular file")
        sys.exit(2)

    with open(yamlpath) as f:
        cofi_spec = None
        try:
            cofi_spec = yaml.load(f, Loader=yaml.FullLoader)
        except e:
            print(
                f"Could not parse specified yaml imput file {yamlpath}. Fix your"
                " specification"
            )
            print("")
            print("")
            print(
                "Underlying exception follows, in case that is helpful in fixing your"
                " yaml file:"
            )
            print("")
            print(e)
            sys.exit(3)
        # TODO need to do some proper checking of the YAML file here.
        # This is left as an exercise for the reader ;-)
        experiment_name = cofi_spec["name"]

        # add path to user module to python module search path, and then load the module
        # TODO, would need to be careful here to check that the user does not have a module
        # name that already exists in our module search path (such as an inbuilt python module).
        print("Loading user module")
        codepath = cofi_spec["fwd_code"]["location"]
        modulename = cofi_spec["fwd_code"]["name"]
        sys.path.append(codepath)
        usermodule = importlib.import_module(modulename)

        if "cofi_init" not in dir(usermodule) or "cofi_misfit" not in dir(usermodule):
            print(
                f"Specified user module/code {modulename} at {codepath} does not"
                " contain mandatory functions cofi_init and cofi_misfit"
            )
            sys.exit(4)

        init_args = cofi_spec["init_info"]
        print(f"Calling user initialization function cofi_init() with args {init_args}")
        usermodule.cofi_init(**init_args)

        # initialize the model to be used in inversion
        m = Model.init_from_yaml(cofi_spec["model"])

        # initialize the inverter
        clsnm = cofi_spec["method"]["name"]
        klass = getattr(cofi_core, clsnm)
        # model and forward should be converted to proper python objects
        cofi_spec["method"]["args"]["model"] = Model.init_from_yaml(
            cofi_spec["method"]["args"]["model"]
        )
        cofi_spec["method"]["args"]["forward"] = usermodule.cofi_misfit
        inv = klass(**cofi_spec["method"]["args"])

        # run it!
        res = inv.run()

        # TODO hack for the moment... convert out model back into something that can be tured
        # into valid YAML. Better will be to create yaml reader/writer method for each object
        # with a YAML decorator.
        if "model" in res:
            res["model"] = res["model"].to_yamlizable()
        for k, v in res.items():
            if isinstance(v, np.ndarray):
                res[k] = v.tolist()

        # dump the results out to YAML
        outfile = experiment_name + "_cofi_out.yml"
        with open(outfile, "w") as outf:
            outf.write(yaml.dump(res))
        print(f"Saved results of inversion to {outfile}")

import argparse
import os
import yaml
import sys
import importlib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='  CoFI main launcher.')
    parser.add_argument('PATH_TO_YAML', type=str, required=True,
                    help='Path to a yaml file specifying your problem')

    args = parser.parse_args()
    yamlpath = args.PATH_TO_YAML
    if not os.path.exists(yamlpath):
        print(f"Speciified yaml file {yamlpath} does not exist")
        sys.exit(1)
    elif os.path.isfile(yamlpath):
        print(f"Speciified yaml path {yamlpath} is not a regular file")
        sys.exit(2)
    
    with open(yamlpath) as file:
        cofi_spec = None
        try: 
            cofi_spec = yaml.load(file, Loader=yaml.FullLoader)
        except e:
            print(f"Could not parse specified yaml imput file {yamlpath}. Fix your specification")
            print("")
            print("")
            print("Underlying exception follows, in case that is helpful in fixing your yaml file:")
            print("")
            print(e)
            sys.exit(3)
        # TODO need to do some proper checking of the YAML file here.
        # This is left as an exercise for the reader ;-)

        # add path to user module to python module search path
        print("Loading user module")
        codepath = cofi_spec["code"]["location"]
        modulename = cofi_spec["code"]["name"]
        sys.path.append(codepath)
        usermodule = importlib.import_module(modulename)

        if "cofi_init" not in dir(usermodule) or "cofi_misfit" not in dir(usermodule):
            print(f"Specified user module/code {} at {cofi_spec["code"]["location"]}")
            sys.exit(4)
        usermodule.cofi_init()






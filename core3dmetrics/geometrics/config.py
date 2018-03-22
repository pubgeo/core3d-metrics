# PROCESS GEOMETRICS CONFIGURATION FILE

import os
import configparser
import json
import glob
import collections
import jsonschema
import pkg_resources
import ast

# module/package name
resource_package = __name__


# HELPER: Locate absolute file path in dict via GLOB
def findfiles(data,path=None):

    for key,file in data.items():
        if not key.lower().endswith('filename'): continue

        print('Searching for "{}"'.format(key))

        if file is None:
            print('  No file specified')
            continue

        # absolute path to file
        if not os.path.isabs(file):
            if path: file = os.path.join(path,file)
            file = os.path.abspath(file)

        # locate file (use glob to allow wildcards)
        files = glob.glob(file)

        if not files:
            print("  WARNING: unable to locate file <{}>".format(file))
            file = None
        else:
            if len(files) > 1:
                print('  WARNING: multiple files located for <{}>, using 1st file'.format(file))

            file = files[0]
            print('  File located <{}>'.format(file))

        # save file to data
        data[key] = file

    return data


# PARSE CONFIGURATION FILE
def parse_config(configfile,refpath=None,testpath=None):

    print('\n=====CONFIGURATION=====')

    # check inputs
    if configfile and not os.path.isfile(configfile):
        raise IOError("Configuration file does not exist")

    if refpath and not os.path.isdir(refpath):
        raise IOError('"refpath" not a valid folder <{}>'.format(refpath))

    if testpath and not os.path.isdir(refpath):
        raise IOError('"testpath" not a valid folder <{}>'.format(testpath))

    # create schema validator object (& check schema itself)
    schema = json.loads(pkg_resources.resource_string(
        resource_package, 'config_schema.json').decode('utf-8'))
    validator = jsonschema.Draft4Validator(schema)
    validator.check_schema(schema)
    
    # load user configuration
    print("\nReading configuration from <{}>".format(configfile))

    # JSON parsing
    if configfile.endswith(('.json','.JSON')):

        # open & read JSON file
        with open(configfile,'r') as fid:
            config = json.load(fid)

    # CONFIG parsing
    elif configfile.endswith(('.config','.CONFIG')):

        # setup config parser
        parser = configparser.ConfigParser()
        parser.optionxform = str # maintain case-sensitive items

        # read entire configuration file into dict
        if len(parser.read(configfile)) == 0:
            raise IOError("Unable to read selected .config file")
        config = {s:dict(parser.items(s)) for s in parser.sections()}   

        # special section/item parsing
        s = 'INPUT.REF'; i = 'CLSMatchValue'; config[s][i] = ast.literal_eval(config[s][i])
        s = 'INPUT.TEST'; i = 'CLSMatchValue'
        if i in config[s]: # Optional Field
            config[s][i] = ast.literal_eval(config[s][i])
        else:
            config[s][i] = config['INPUT.REF'][i]

        # bool(config[s][i]) does not interpret 'true'/'false' strings
        s = 'OPTIONS'; i = 'QuantizeHeight'; config[s][i] = parser.getboolean(s,i)  
        s = 'PLOTS'; i = 'ShowPlots'; config[s][i] = parser.getboolean(s,i) 
        s = 'PLOTS'; i = 'SavePlots'; config[s][i] = parser.getboolean(s,i)
        s = 'MATERIALS.REF'; i = 'MaterialNames'; config[s][i] = config[s][i].split(',')
        s = 'MATERIALS.REF'; i = 'MaterialIndicesToIgnore'; config[s][i] = [int(v) for v in config[s][i].split(',')]

    # unrecognized config file type
    else:
        raise IOError('Unrecognized configuration file')


    # locate files for each "xxxFilename" configuration parameter
    # this makes use of "refpath" and "testpath" arguments for relative filenames
    # we do this before validation to ensure required files are located
    for item in [('INPUT.REF',refpath),('INPUT.TEST',testpath)]:
        sec = item[0]; path = item[1]
        print('\nPROCESSING "{}" FILES'.format(sec))
        config[sec] = findfiles(config[sec],path)


    # validate final configuration against schema
    try:
        validator.validate(config)
        print('\nCONFIGURATION VALIDATED')

    except jsonschema.exceptions.ValidationError:
        print('\n*****INVALID CONFIGURATION FILE*****\n')
        for error in sorted(validator.iter_errors(config), key=str):
            print('ERROR: {}\n'.format(error))

        raise jsonschema.exceptions.ValidationError('validation error')

  
    # for easier explotation, ensure some configuration options are tuple/list
    opts = (('INPUT.TEST','CLSMatchValue'),('INPUT.REF','CLSMatchValue'),
        ('MATERIALS.REF','MaterialIndicesToIgnore'))

    for opt in opts:
        s = opt[0]; i = opt[1];
        try:
            _ = (v for v in config[s][i])
        except:
            config[s][i] = [config[s][i]]


    # print final configuration
    print('\nFINAL CONFIGURATION')
    print(json.dumps(config,indent=2))

    # cleanup
    return config

#!/usr/bin/env python3
#
# This script generates an INGRID input YAML file from a GEQDSK equilibrium
#
# Example
#
#     ./geqdsk_ingrid neqdsk -o neqdsk.yaml
#

import edgegridanalysis
from argparse import ArgumentParser

def main(*_args):
    parser = ArgumentParser()
    parser.add_argument("inputfile", help="GEQDSK file")
    parser.add_argument('-o', '--output', default="ingrid.yaml", help="YAML file for INGRID")
    args = parser.parse_args()

    # Read equilibrium from input file
    eq = edgegridanalysis.equilibrium.read_geqdsk(args.inputfile)

    # Calculate equilibrium type
    eq_type = edgegridanalysis.equilibrium.EquilibriumType(eq)

    # Convert to dict and then yaml
    with open(args.output, 'w') as f:
        f.write(edgegridanalysis.output.to_yaml(eq_type.to_ingrid_dict()))

if __name__ == "__main__":
    main()


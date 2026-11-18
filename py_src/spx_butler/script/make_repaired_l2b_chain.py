#!/usr/bin/env python3
import argparse

from lsst.daf.butler import Butler, CollectionType


def main():
    """Driver to prepend retry run ahead of original L2b run in a CHAINED collection."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--butler", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--retry-run", required=True)
    parser.add_argument("--original-run", required=True)
    args = parser.parse_args()

    butler = Butler(args.butler, writeable=True)
    reg = butler.registry

    try:
        reg.getCollectionType(args.chain)
    except Exception:
        reg.registerCollection(args.chain, type=CollectionType.CHAINED)

    reg.setCollectionChain(args.chain, [args.retry_run, args.original_run])
    print(f"Created/updated chain {args.chain} = [{args.retry_run}, {args.original_run}]")


if __name__ == "__main__":
    main()

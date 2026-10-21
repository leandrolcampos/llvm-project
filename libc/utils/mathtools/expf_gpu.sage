# Print float-float lookup table entries for 2^(i / table_size) in {lo, hi} order.
# Run with: sage libc/utils/mathtools/expf_gpu.sage -- --table-size 2
# With --rounding-bit, the low bit of hi stores a residual sign and must be
# cleared before hi is used in arithmetic.

import argparse
import sys

from sage.all import QQ, ZZ, RealField


def get_c_fp_literal_suffix(x):
    suffixes = {
        11: "f16",
        24: "f",
        53: "",
        64: "L",
    }

    prec = x.prec()

    try:
        return suffixes[prec]
    except KeyError:
        raise ValueError(
            f"no C floating-point literal suffix for precision {prec}"
        ) from None


def get_hex(x):
    if x < 0:
        return "-" + get_hex(-x)

    hex_str = x.hex()
    leading_digit = hex_str[2]

    if leading_digit in ["0", "1"]:
        exponent_adjustment = 0
    elif leading_digit in ["2", "3"]:
        hex_str = (x / 2).hex()
        exponent_adjustment = 1
    elif leading_digit in ["4", "5", "6", "7"]:
        hex_str = (x / 4).hex()
        exponent_adjustment = 2
    else:
        hex_str = (x / 8).hex()
        exponent_adjustment = 3

    significand_str, exponent_str = hex_str.split("p")
    exponent = ZZ(exponent_str) + exponent_adjustment
    exponent_str = "+" + exponent.str() if exponent >= 0 else exponent.str()

    return significand_str + "p" + exponent_str + get_c_fp_literal_suffix(x)


def get_default_work_prec(prec):
    work_precs = {
        11: 50,
        24: 100,
        53: 200,
        64: 250,
    }

    return work_precs.get(prec, 4 * prec)


def split(x, *, num_words, prec, work_prec=None):
    if num_words < 1:
        raise ValueError("num_words must be at least 1")

    if work_prec is None:
        work_prec = get_default_work_prec(prec)

    if work_prec < prec:
        raise ValueError("work_prec must be at least prec")

    R = RealField(prec)

    def split_at(p):
        residual = x.n(prec=p)
        words = []

        for _ in range(num_words):
            word = R(residual)
            words.append(word)
            residual -= word.exact_rational()

        if num_words > 1:
            last_word = words[-1]
            at_boundary = abs(last_word) == words[-2].ulp() / 2
            if residual != 0 and (at_boundary or last_word == 0):
                if residual > 0:
                    words[-1] = last_word.nextabove()
                else:
                    words[-1] = last_word.nextbelow()

        return tuple(words)

    result = split_at(work_prec)
    check = split_at(2 * work_prec)

    if result != check:
        raise ValueError("split is not stable; increase work_prec")

    return result


def generate_lookup_table(func, *, num_entries, **split_kwargs):
    return tuple(
        split(func(QQ(i) / num_entries), **split_kwargs) for i in range(num_entries)
    )


def split_with_rounding_bit(x, *, work_prec=200):
    R = RealField(24)
    H = RealField(23)

    def split_at(p):
        residual = x.n(prec=p)
        hi = R(H(residual))
        residual -= hi.exact_rational()
        lo = R(residual)
        residual -= lo.exact_rational()

        # Rounding hi to 23 bits leaves its low significand bit free. Set it
        # when the exact table value exceeds the sum of the decoded words.
        if residual > 0:
            hi = hi.nextabove()
        return hi, lo

    result = split_at(work_prec)
    if result != split_at(2 * work_prec):
        raise ValueError("split is not stable; increase work_prec")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Print float-float lookup table entries in {lo, hi} order."
    )
    parser.add_argument(
        "--table-size", type=int, default=2, help="number of entries (default: 2)"
    )
    parser.add_argument(
        "--rounding-bit",
        action="store_true",
        help="encode a residual sign in the low bit of a 23-bit high word",
    )
    # Sage may leave the separator used to forward script arguments in sys.argv.
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    args = parser.parse_args(argv)
    if args.table_size < 1:
        parser.error("--table-size must be at least 1")

    if args.rounding_bit:
        table = tuple(
            split_with_rounding_bit(ZZ(2)**(QQ(i) / args.table_size))
            for i in range(args.table_size)
        )
    else:
        table = generate_lookup_table(
            lambda x: ZZ(2)**x, num_entries=args.table_size, num_words=2, prec=24
        )

    for hi, lo in table:
        print("    {" + get_hex(lo) + ", " + get_hex(hi) + "},")


main()

# Print lookup table entries for 2^(i / table_size).
# Run with: sage libc/utils/mathtools/expf_gpu.sage -- --table-size 2
# Additive entries reconstruct as sum(words); two-word multiplicative entries
# reconstruct as words[0] * (1 + words[1]). One-word entries store only words[0].
# Word precisions and rounding modes are specified from high to low.
# Use --word-order low-first for FloatFloat initializers in {lo, hi} order.
# With --with-rounding-bit, clear the low bit of the penultimate word before
# using it in arithmetic. This refers to the word order before output reversal.
# Use --no-suffix to omit C literal suffixes and allow arbitrary precisions.

import argparse
import sys

from sage.all import Integer, QQ, RealField, ZZ


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


def get_hex(x, *, with_suffix=True):
    if x < 0:
        return "-" + get_hex(-x, with_suffix=with_suffix)

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

    suffix = get_c_fp_literal_suffix(x) if with_suffix else ""
    return significand_str + "p" + exponent_str + suffix


def get_default_work_prec(prec):
    work_precs = {
        11: 50,
        24: 100,
        53: 200,
        64: 250,
    }

    return work_precs.get(prec, 4 * prec)


def split(
    x,
    *,
    num_words,
    prec,
    work_prec=None,
    rounding_mode="RNDN",
    with_rounding_bit=False,
    split_type="additive",
):
    """Split x into additive words or a main word and a relative correction.

    Words are returned from high to low. With a rounding bit, decode the
    penultimate word before reconstruction; the bit indicates whether x is
    greater than the reconstructed value (1) or not (0).
    """
    integer_types = (int, Integer)

    if not isinstance(num_words, integer_types) or num_words < 1:
        raise ValueError("num_words must be a positive integer")

    if split_type not in ("additive", "multiplicative"):
        raise ValueError("split_type must be additive or multiplicative")
    if split_type == "multiplicative" and num_words > 2:
        raise ValueError("multiplicative split requires num_words <= 2")

    if isinstance(prec, integer_types):
        prec = [prec] * num_words
    elif isinstance(prec, list):
        prec = prec.copy()
    else:
        raise TypeError("prec must be an integer or a list of integers")

    if len(prec) != num_words:
        raise ValueError("prec must have num_words entries")

    if isinstance(rounding_mode, str):
        rounding_mode = [rounding_mode] * num_words
    elif isinstance(rounding_mode, list):
        rounding_mode = rounding_mode.copy()
    else:
        raise TypeError("rounding_mode must be a string or a list of strings")

    if len(rounding_mode) != num_words:
        raise ValueError("rounding_mode must have num_words entries")

    fields = [
        RealField(p, rnd=mode)
        for p, mode in zip(prec, rounding_mode)
    ]
    prec = [field.prec() for field in fields]

    if with_rounding_bit:
        if num_words <= 1:
            raise ValueError("with_rounding_bit requires num_words > 1")
        if prec[-2] <= 1:
            raise ValueError(
                "with_rounding_bit requires penultimate-word precision > 1"
            )

    if work_prec is None:
        work_prec = max(
            get_default_work_prec(max(prec)),
            sum(prec) * 2,
        )
    elif not isinstance(work_prec, integer_types):
        raise TypeError("work_prec must be an integer or None")

    if work_prec < max(prec):
        raise ValueError("work_prec must be at least every word precision")

    rounding_fields = fields.copy()
    if with_rounding_bit:
        rounding_fields[-2] = RealField(
            prec[-2] - 1,
            rnd=rounding_mode[-2],
        )

    def split_at(p):
        residual = x.n(prec=p)
        words = []

        for i, (field, rounding_field) in enumerate(zip(fields, rounding_fields)):
            # Conversion to the storage precision is exact.
            word = field(rounding_field(residual))
            words.append(word)
            residual -= word.exact_rational()

            if split_type == "multiplicative" and num_words == 2 and i == 0:
                if word == 0:
                    if residual != 0:
                        raise ValueError(
                            "multiplicative split requires a nonzero main word "
                            "for a nonzero value"
                        )
                else:
                    # Subtract before dividing to retain the small correction.
                    residual /= word.exact_rational()

        # Keep directed low-word rounding unchanged at boundaries.
        # This boundary condition applies only to additive words.
        if (
            split_type == "additive"
            and num_words > 1
            and rounding_mode[-1] == "RNDN"
            and not with_rounding_bit
        ):
            last_word = words[-1]
            at_boundary = abs(last_word) == words[-2].ulp() / 2

            if residual != 0 and (at_boundary or last_word == 0):
                if residual > 0:
                    words[-1] = last_word.nextabove()
                else:
                    words[-1] = last_word.nextbelow()

        # A relative residual has the opposite reconstruction-error sign when
        # the main word is negative. Encode the error in the original scale.
        positive_error = residual > 0
        if split_type == "multiplicative" and num_words == 2 and words[0] < 0:
            positive_error = residual < 0

        if with_rounding_bit and positive_error:
            word = words[-2]

            if word == 0:
                raise ValueError(
                    "cannot encode a positive residual sign in a zero word"
                )

            # Set the reserved significand bit by moving away from zero.
            words[-2] = (
                word.nextabove() if word > 0 else word.nextbelow()
            )

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


def main():
    parser = argparse.ArgumentParser(
        description="Print lookup table entries for 2^(i / table_size).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--table-size", type=int, default=2, help="number of table entries"
    )
    parser.add_argument(
        "--num-words", type=int, default=2, help="number of words per entry"
    )
    parser.add_argument(
        "--split-type",
        choices=("additive", "multiplicative"),
        default="additive",
        help="additive: sum(words); multiplicative: hi * (1 + lo), at most two words",
    )
    parser.add_argument(
        "--prec",
        type=int,
        nargs="+",
        default=[24],
        metavar="BITS",
        help="one precision for all words, or one per word, from high to low",
    )
    parser.add_argument(
        "--work-prec",
        type=int,
        default=None,
        metavar="BITS",
        help="working precision; omit to select it automatically",
    )
    parser.add_argument(
        "--rounding-mode",
        nargs="+",
        default=["RNDN"],
        metavar="MODE",
        help="one Sage rounding mode for all words, or one per word, from high to low",
    )
    parser.add_argument(
        "--with-rounding-bit",
        action="store_true",
        help="reserve the low bit of the penultimate word for the residual sign",
    )
    parser.add_argument(
        "--word-order",
        choices=("high-first", "low-first"),
        default="high-first",
        help="output order; does not change the order of --prec or --rounding-mode",
    )
    parser.add_argument(
        "--no-suffix",
        action="store_true",
        help="omit C literal suffixes from the hexadecimal output",
    )
    # Sage may leave the separator used to forward script arguments in sys.argv.
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    args = parser.parse_args(argv)
    if args.table_size < 1:
        parser.error("--table-size must be at least 1")

    prec = args.prec[0] if len(args.prec) == 1 else args.prec
    rounding_mode = (
        args.rounding_mode[0]
        if len(args.rounding_mode) == 1
        else args.rounding_mode
    )
    try:
        table = generate_lookup_table(
            lambda x: ZZ(2)**x,
            num_entries=args.table_size,
            num_words=args.num_words,
            prec=prec,
            work_prec=args.work_prec,
            rounding_mode=rounding_mode,
            with_rounding_bit=args.with_rounding_bit,
            split_type=args.split_type,
        )
        entries = []
        for words in table:
            if args.word_order == "low-first":
                words = reversed(words)
            literals = (
                get_hex(word, with_suffix=not args.no_suffix)
                for word in words
            )
            entries.append("    {" + ", ".join(literals) + "},")
    except (TypeError, ValueError) as error:
        parser.error(str(error))

    print("\n".join(entries))


main()

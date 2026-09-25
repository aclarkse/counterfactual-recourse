"""Deprecated empirical-group shift plot.

The original plot used observed advantaged true negatives as its reference,
which does not match the revised recipient-Z estimand. It is retained as an
explicitly disabled command so stale scripts fail loudly and descriptively.
"""


def main():
    raise SystemExit(
        "This legacy plot is disabled because its reference population does "
        "not match the revised estimand. Run evaluation.ablate_recourse and "
        "plot mediated_post against transport from its CSV output."
    )


if __name__ == "__main__":
    main()

from pipeline.steps.runtime import run_runtime


def main():
    try:
        run_runtime(use_real_gantry=True, execute_targets=True)
        print("\n[MAIN] Runtime finished.")
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user.")
    except Exception as exc:
        print(f"\n[MAIN] Runtime failed: {exc}")
        raise


if __name__ == "__main__":
    main()

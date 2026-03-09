def plan_targets(matched_targets):
    print("\n=== PLAN ===")

    if not matched_targets:
        print("No matched targets to plan.")
        return []

    target_queue = sorted(
        matched_targets,
        key=lambda t: t["score"],
        reverse=True,
    )

    print(f"Planned {len(target_queue)} targets.")
    return target_queue
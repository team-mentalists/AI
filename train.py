                    stop(f"Non-numeric label value in {label_path}, line {line_number}.")

                if old_id in CLASS_MAPPING:
                    new_id = CLASS_MAPPING[old_id]
                elif 0 <= old_id < len(CLASS_NAMES):
                    new_id = old_id
                else:
                    stop(
                        f"Unknown class ID {old_id} in {label_path}. "
                        "Expected IDs 0-6 or 15-21."
                    )

                if any(v < 0 or v > 1 for v in coords):
                    stop(
                        f"YOLO coordinates must be between 0 and 1: "
                        f"{label_path}, line {line_number}"
                    )

                converted_lines.append(
                    " ".join([str(new_id)] + [str(v) for v in coords])
                )
                total_objects[new_id] += 1

            (dst_labels / label_path.name).write_text(
                "\n".join(converted_lines) + ("\n" if converted_lines else ""),
                encoding="utf-8",
            )

        split_counts[split] = len(images)
        print(
            f"{split}: {len(images)} images, "
            f"{len(label_files)} label files copied"
        )

    if "train" not in split_counts or "validation" not in split_counts:
        stop("Both train/ and validation/ splits are required.")

    data_yaml = PREPARED / "data.yaml"
    config = {
        "path": str(PREPARED.resolve()),
        "train": "train/images",
        "val": "validation/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    if "test" in split_counts:
        config["test"] = "test/images"

    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    print("\nClass counts:", dict(sorted(total_objects.items())))
    print("Dataset YAML:", data_yaml)
    return data_yaml


def main():
    print("Project folder:", ROOT)
    print("Dataset folder:", SOURCE)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("No CUDA GPU found; training on CPU will be slower.")
        device = "cpu"

    data_yaml = prepare_dataset()

    print(f"\nLoading model: {MODEL_WEIGHTS}")
    # The first run may download this pretrained weights file.
    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=0,              # Safer on Windows
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        amp=torch.cuda.is_available(),
        cache=False,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        project=str(RUNS),
        name=RUN_NAME,
        exist_ok=True,
    )

    best_path = RUNS / RUN_NAME / "weights" / "best.pt"
    if not best_path.is_file():
        stop(f"Training finished but best.pt was not found at {best_path}")

    project_best = ROOT / "best.pt"
    shutil.copy2(best_path, project_best)
    print("\nBest model copied to:", project_best)

    # Evaluate on test split only when it exists.
    if (PREPARED / "test" / "images").is_dir():
        print("\nEvaluating on test split...")
        best_model = YOLO(str(project_best))
        metrics = best_model.val(
            data=str(data_yaml),
            split="test",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=device,
        )
        print("\n========== TEST RESULTS ==========")
        print("mAP50:", metrics.box.map50)
        print("mAP50-95:", metrics.box.map)
        print("Precision:", metrics.box.mp)
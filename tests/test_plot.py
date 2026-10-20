"""The plotting CLI, on small metrics files written here."""

import json

import pytest

from generator.plot import load_run, main


@pytest.fixture
def metrics_dir(tmp_path):
    directory = tmp_path / "metrics"
    directory.mkdir()

    def write(name, rows):
        (directory / name).write_text("".join(json.dumps(row) + "\n" for row in rows))

    write("a.jsonl", [{"epoch": e, "step": e, "train_loss": 1 / e, "validation_loss": 1.1 / e} for e in (1, 2)])
    summary = {"classifier_accuracy": 0.5, "classifier_confidence": 0.6, "classifier_target_probability": 0.4}
    write("b.jsonl", [{"epoch": e, "step": e, "train_loss": 0.4, "validation_loss": 0.5} | summary for e in (2, 3)])
    samples = [
        {
            "epoch": e,
            "target": d,
            "sample": 0,
            "predicted": d,
            "correct": True,
            "confidence": 0.9,
            "target_probability": 0.9,
        }
        for e in (2, 3)
        for d in range(10)
    ]
    write("b_classifier.jsonl", samples)
    return directory


def test_joined_runs_take_the_later_epoch(metrics_dir):
    run = load_run("a+b", metrics_dir)
    assert run.series("train_loss") == ([1, 2, 3], [1.0, 0.4, 0.4])
    assert run.sample_epochs() == [2, 3]


@pytest.mark.parametrize(
    ("args", "files"),
    [
        ([], ["a+b.png"]),
        (["-p", "loss", "-p", "best_validation_loss", "--format", "pdf"], None),
        (["--separate", "--format", "svg"], ["a+b-loss.svg", "a+b-confusion-e003.svg", "a+b-digit-accuracy.svg"]),
    ],
)
def test_writes_figures(metrics_dir, tmp_path, args, files):
    output = tmp_path / "plots" / "a+b"
    if files is None:  # best_validation_loss is not logged by these runs
        with pytest.raises(SystemExit, match="Unknown plot"):
            main(["--run", "a+b", "--metrics-dir", str(metrics_dir), "-o", str(output), *args])
        return
    main(["--run", "a+b", "--metrics-dir", str(metrics_dir), "-o", str(output), *args])
    for name in files:
        assert (tmp_path / "plots" / name).stat().st_size > 0


def test_compares_runs_with_a_panel_each_for_samples(metrics_dir, tmp_path):
    main(["--run", "a", "--run", "b", "--metrics-dir", str(metrics_dir), "-o", str(tmp_path / "x"), "--separate"])
    names = sorted(p.name for p in tmp_path.glob("x-*.png"))
    assert names == [
        "x-accuracy.png",
        "x-confidence.png",
        "x-confusion-b-e003.png",
        "x-digit-accuracy-b.png",
        "x-loss.png",
    ]


def test_digit_bars_one_file_per_epoch(metrics_dir, tmp_path):
    output = str(tmp_path / "x")
    main(["--run", "b", "--metrics-dir", str(metrics_dir), "-o", output, "-p", "digit-bars", "--separate"])
    assert sorted(p.name for p in tmp_path.glob("x-*.png")) == ["x-digit-bars-e002.png", "x-digit-bars-e003.png"]
    main(["--run", "b", "--metrics-dir", str(metrics_dir), "-o", output, "-p", "digit-bars", "--epoch", "3"])
    assert (tmp_path / "x.png").exists()
    with pytest.raises(SystemExit, match="no classifier samples for epoch 7"):
        main(["--run", "b", "--metrics-dir", str(metrics_dir), "-o", output, "-p", "confusion", "--epoch", "7"])


@pytest.mark.parametrize("value", ["accuracy", "confidence", "target-probability"])
def test_bar_values(metrics_dir, tmp_path, value):
    for panel in ("digit-bars", "digit-bars-stacked"):
        output = tmp_path / f"{panel}-{value}.svg"
        main(["--run", "b", "--metrics-dir", str(metrics_dir), "-o", str(output), "-p", panel, "--bar-value", value])
        assert output.stat().st_size > 0


def test_stacked_bars_use_the_chosen_epochs(metrics_dir, tmp_path):
    output = tmp_path / "stacked.svg"
    args = ["--run", "b", "--metrics-dir", str(metrics_dir), "-o", str(output), "-p", "digit-bars-stacked"]
    main([*args, "--epoch", "3", "2"])
    assert "epochs 2, 3" in output.read_text()
    with pytest.raises(SystemExit, match="no classifier samples for epoch 1"):
        main([*args, "--epoch", "1", "3"])

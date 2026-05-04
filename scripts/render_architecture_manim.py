from __future__ import annotations

from pathlib import Path
import sys

from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    RIGHT,
    UP,
    AnimationGroup,
    Circle,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    FadeTransform,
    Group,
    LaggedStart,
    Line,
    MovingCameraScene,
    RoundedRectangle,
    Succession,
    SurroundingRectangle,
    Text,
    VGroup,
    config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from cnn_xai_lab.config import OUTPUTS_DIR


BG = "#09070f"
PANEL = "#181126"
PANEL_2 = "#211633"
PURPLE = "#8f5cff"
PURPLE_SOFT = "#d2b8ff"
TEXT = "#f5efff"
TEXT_SOFT = "#b7abd6"
LINE = "#4d2d82"
GLOW = "#c97dff"


config.background_color = BG
config.pixel_width = 1280
config.pixel_height = 720
config.frame_rate = 30


class ArchitectureFlowScene(MovingCameraScene):
    def make_stage(self, title: str, subtitle: str, shape: str, width: float = 1.6, height: float = 2.9) -> VGroup:
        panel = RoundedRectangle(
            corner_radius=0.18,
            width=width,
            height=height,
            stroke_color=PURPLE,
            stroke_width=1.8,
            fill_color=PANEL,
            fill_opacity=0.96,
        )
        cap = RoundedRectangle(
            corner_radius=0.14,
            width=width * 0.84,
            height=0.38,
            stroke_width=0,
            fill_color=PANEL_2,
            fill_opacity=1.0,
        ).move_to(panel.get_top() + DOWN * 0.36)
        title_text = Text(title, font_size=26, color=TEXT, weight="BOLD").move_to(cap)
        subtitle_text = Text(subtitle, font_size=17, color=TEXT_SOFT).next_to(cap, DOWN, buff=0.28)
        shape_text = Text(shape, font_size=19, color=PURPLE_SOFT, weight="BOLD").next_to(
            subtitle_text, DOWN, buff=1.06
        )
        glyph = RoundedRectangle(
            corner_radius=0.12,
            width=width * 0.56,
            height=0.84,
            stroke_color=LINE,
            stroke_width=1.2,
            fill_color=PURPLE,
            fill_opacity=0.12,
        ).next_to(shape_text, UP, buff=0.42)
        return VGroup(panel, cap, title_text, subtitle_text, glyph, shape_text)

    def make_input_panel(self) -> VGroup:
        panel = RoundedRectangle(
            corner_radius=0.2,
            width=1.95,
            height=3.4,
            stroke_color=PURPLE,
            stroke_width=2.0,
            fill_color=PANEL,
            fill_opacity=0.98,
        )
        title = Text("Input", font_size=28, color=TEXT, weight="BOLD").next_to(panel.get_top(), DOWN, buff=0.34)
        subtitle = Text("Imagen RGB", font_size=18, color=TEXT_SOFT).next_to(title, DOWN, buff=0.12)
        rows = []
        for row_index in range(4):
            row = []
            for col_index in range(4):
                cell = RoundedRectangle(
                    corner_radius=0.03,
                    width=0.18,
                    height=0.18,
                    stroke_width=0.8,
                    stroke_color=LINE,
                    fill_opacity=1.0,
                    fill_color=["#8f5cff", "#d86fff", "#5ac8fa"][(row_index + col_index) % 3],
                )
                row.append(cell)
            rows.append(VGroup(*row).arrange(RIGHT, buff=0.05))
        grid = VGroup(*rows).arrange(DOWN, buff=0.05).move_to(panel.get_center() + DOWN * 0.2)
        shape = Text("224 x 224 x 3", font_size=20, color=PURPLE_SOFT, weight="BOLD").next_to(
            grid, DOWN, buff=0.34
        )
        return VGroup(panel, title, subtitle, grid, shape)

    def make_stage_specs(self) -> list[tuple[str, str, str]]:
        return [
            ("Conv2D", "32 filtros | kernel 5x5 | ReLU", "224 x 224 x 32"),
            ("BatchNorm", "Normaliza activaciones", "224 x 224 x 32"),
            ("MaxPool", "Reduce resolucion", "112 x 112 x 32"),
            ("Conv2D", "64 filtros | kernel 5x5 | ReLU", "112 x 112 x 64"),
            ("BatchNorm", "Estabilidad intermedia", "112 x 112 x 64"),
            ("MaxPool", "Compresion espacial", "56 x 56 x 64"),
            ("Conv2D", "128 filtros | kernel 5x5 | ReLU", "56 x 56 x 128"),
            ("BatchNorm", "Mapa mas profundo", "56 x 56 x 128"),
            ("MaxPool", "Rasgos condensados", "28 x 28 x 128"),
            ("Conv2D", "192 filtros | kernel 5x5 | ReLU", "28 x 28 x 192"),
            ("GAP", "GlobalAveragePooling2D", "192"),
            ("Dense", "96 neuronas | ReLU", "96"),
            ("Dropout", "Regularizacion 0.50", "96"),
            ("Salida", "Sigmoid binaria", "1"),
        ]

    def construct(self) -> None:
        self.camera.background_color = BG

        input_panel = self.make_input_panel()
        stages = [self.make_stage(*spec, width=1.48, height=3.0) for spec in self.make_stage_specs()]
        stage_group = VGroup(*stages).arrange(RIGHT, buff=0.36, aligned_edge=UP)
        stage_group.next_to(input_panel, RIGHT, buff=0.52)

        pipeline = VGroup(input_panel, stage_group).scale(0.88)
        pipeline.move_to(ORIGIN + DOWN * 0.35)

        connectors = VGroup()
        previous = input_panel
        for stage in stages:
            link = Line(
                previous.get_right() + RIGHT * 0.03,
                stage.get_left() + LEFT * 0.03,
                stroke_color=LINE,
                stroke_width=2.0,
            )
            connectors.add(link)
            previous = stage

        self.camera.frame.set(width=7.6)
        self.camera.frame.move_to(VGroup(input_panel, stages[0], stages[1]).get_center())

        self.play(FadeIn(input_panel, shift=RIGHT * 0.15), run_time=1.0)
        self.play(
            LaggedStart(*[FadeIn(stage, shift=UP * 0.08) for stage in stages], lag_ratio=0.08),
            Create(connectors),
            run_time=3.2,
        )

        traveler = Dot(input_panel.get_right() + RIGHT * 0.08, color=GLOW, radius=0.14)
        spotlight = SurroundingRectangle(input_panel, color=PURPLE_SOFT, buff=0.08, corner_radius=0.18)
        callout = Text("Tensor de entrada 224 x 224 x 3", font_size=22, color=TEXT).next_to(
            input_panel, DOWN, buff=1.15
        )

        self.play(FadeIn(traveler, scale=0.6), Create(spotlight), FadeIn(callout, shift=UP * 0.1), run_time=0.7)

        previous_text = callout
        chunk_groups = [
            VGroup(input_panel, *stages[:3]),
            VGroup(stages[2], stages[3], stages[4], stages[5]),
            VGroup(stages[5], stages[6], stages[7], stages[8]),
            VGroup(stages[8], stages[9], stages[10], stages[11]),
            VGroup(stages[11], stages[12], stages[13]),
        ]
        chunk_map = {
            0: 0,
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
            9: 3,
            10: 3,
            11: 3,
            12: 4,
            13: 4,
        }
        active_chunk = 0

        for stage_index, (stage, (name, subtitle_text, shape_text), connector) in enumerate(
            zip(stages, self.make_stage_specs(), connectors)
        ):
            target_chunk = chunk_map[stage_index]
            if target_chunk != active_chunk:
                focus_group = chunk_groups[target_chunk]
                self.play(
                    self.camera.frame.animate.set(width=focus_group.width + 1.45).move_to(
                        focus_group.get_center()
                    ),
                    run_time=0.95,
                )
                active_chunk = target_chunk

            next_label = Text(
                f"{name}: {subtitle_text}  |  tensor {shape_text}",
                font_size=20,
                color=TEXT,
            ).next_to(stage, DOWN, buff=1.15)
            next_spotlight = SurroundingRectangle(stage, color=PURPLE_SOFT, buff=0.08, corner_radius=0.18)
            self.play(
                traveler.animate.move_to(connector.get_center()),
                run_time=0.42,
            )
            self.play(
                traveler.animate.move_to(stage.get_left() + RIGHT * 0.15),
                FadeOut(previous_text, shift=UP * 0.06),
                FadeIn(next_label, shift=UP * 0.06),
                FadeTransform(spotlight, next_spotlight),
                run_time=0.72,
            )
            pulse = Circle(radius=0.2, color=GLOW, stroke_width=3).move_to(stage.get_center())
            self.play(Create(pulse), FadeOut(pulse), run_time=0.48)
            previous_text = next_label

        output_glow = Circle(radius=0.34, color=GLOW, stroke_width=4).move_to(stages[-1].get_center())
        final_badge = RoundedRectangle(
            corner_radius=0.18,
            width=4.7,
            height=0.9,
            stroke_color=PURPLE,
            stroke_width=1.8,
            fill_color=PANEL_2,
            fill_opacity=0.98,
        ).next_to(stages[-1], UP, buff=0.34)
        final_text = Text("Salida final: probabilidad female vs male", font_size=22, color=TEXT).move_to(
            final_badge
        )
        overview_text = Text(
            "Vista completa de la arquitectura",
            font_size=22,
            color=TEXT,
            weight="BOLD",
        ).next_to(pipeline, DOWN, buff=1.0)

        self.play(
            AnimationGroup(
                Create(output_glow),
                FadeIn(final_badge, shift=UP * 0.12),
                FadeIn(final_text, shift=UP * 0.12),
                lag_ratio=0.1,
            ),
            run_time=1.0,
        )
        self.play(
            self.camera.frame.animate.set(width=pipeline.width + 1.3).move_to(pipeline.get_center()),
            FadeOut(previous_text, shift=UP * 0.06),
            FadeIn(overview_text, shift=UP * 0.06),
            run_time=1.35,
        )
        self.play(Succession(FadeOut(output_glow), FadeIn(output_glow), FadeOut(output_glow)), run_time=1.3)
        self.wait(2.2)


if __name__ == "__main__":
    output_dir = OUTPUTS_DIR / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Render this scene with:")
    print(
        f'python -m manim -q h --disable_caching --media_dir "{output_dir}" "{__file__}" ArchitectureFlowScene'
    )

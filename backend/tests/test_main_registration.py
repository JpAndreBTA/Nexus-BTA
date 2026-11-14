"""No heavy GPU imports: validate app registration order in production module."""
from __future__ import annotations

import ast
import unittest
from pathlib import Path


class MainRegistrationTests(unittest.TestCase):
    def test_fastapi_app_exists_before_first_top_level_registration(self):
        source = Path(__file__).resolve().parents[1] / "nexus_backend" / "main.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        app_creation = None
        app_uses = []
        for node in module.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, ast.Name) and target.id == "app" for target in targets):
                    value = node.value
                    if isinstance(value, ast.Call) and (
                        isinstance(value.func, ast.Name) and value.func.id == "FastAPI"
                    ):
                        app_creation = node.lineno
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name) and call.func.value.id == "app":
                    app_uses.append((node.lineno, call.func.attr))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                        value = decorator.func.value
                        if isinstance(value, ast.Name) and value.id == "app":
                            app_uses.append((node.lineno, "route"))
        self.assertIsNotNone(app_creation, "FastAPI app must be created in main.py")
        self.assertTrue(app_uses, "Test must find real router registrations")
        before = [(line, action) for line, action in app_uses if line <= app_creation]
        self.assertEqual(before, [], "No app method or route decorator may run before app = FastAPI(...)")

    def test_comfy_registration_after_app_creation(self):
        module_path = Path(__file__).resolve().parents[1] / "nexus_backend" / "main.py"
        source = module_path.read_text(encoding="utf-8")
        self.assertLess(source.index('app = FastAPI('),
                        source.index('app.include_router(make_comfy_capability_router('))

"""Скачать и проверить исходные файлы соревнования без анализа данных."""

import csv
import sys
from pathlib import Path

import kagglehub
from kagglehub.exceptions import KaggleApiHTTPError, UnauthenticatedError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
COMPETITION = "sweettv-tv-program-recommender"
EXPECTED_FILES = (
    "dataset11-30.csv",
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
    "export_arh_31-42-final.csv",
    "submission.csv",
)
AUTH_HELP = (
    "Примите правила своим аккаунтом Kaggle: "
    f"https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
    "Создайте API token: https://www.kaggle.com/settings/api\n"
    "Сохраните токен в ~/.kaggle/access_token либо настройте KAGGLE_API_TOKEN. "
    "Также поддерживается ~/.kaggle/kaggle.json (Legacy API Credentials).\n"
    "После настройки повторите запуск скрипта. Не добавляйте токен в код или Git."
)


def check_csv(path: Path) -> int:
    """Проверить непустой файл и чтение заголовка и первой записи."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Отсутствует или пустой CSV: {path}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader)
            first_row = next(reader)
        if not any(header) or not any(first_row):
            raise ValueError("пустой заголовок или первая запись")
    except (UnicodeError, csv.Error, StopIteration, ValueError) as error:
        raise ValueError(f"Не удалось прочитать CSV: {path}. {error}") from error
    return path.stat().st_size


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    missing = []
    for name in EXPECTED_FILES:
        path = DATA_DIR / name
        if path.exists():
            # Существующий повреждённый файл не заменяется автоматически.
            check_csv(path)
        else:
            missing.append(name)

    if missing:
        # kagglehub распаковывает скачанный архив и удаляет его автоматически.
        if not any(DATA_DIR.iterdir()):
            kagglehub.competition_download(COMPETITION, output_dir=str(DATA_DIR))
        else:
            # Для частично заполненной папки скачиваем только недостающие файлы.
            for name in missing:
                kagglehub.competition_download(
                    COMPETITION, path=name, output_dir=str(DATA_DIR)
                )
    else:
        print("Все исходные файлы уже есть; скачивание пропущено.")

    for name in EXPECTED_FILES:
        size = check_csv(DATA_DIR / name)
        print(f"{name}: {size:,} байт ({size / 1024**2:.2f} MiB), CSV читается")
    print("Competition files:", DATA_DIR)


if __name__ == "__main__":
    try:
        main()
    except UnauthenticatedError:
        sys.exit(f"Требуется авторизация Kaggle.\n{AUTH_HELP}")
    except KaggleApiHTTPError as error:
        status = error.response.status_code if error.response is not None else None
        if status in (401, 403):
            sys.exit(f"Kaggle вернул HTTP {status}.\n{AUTH_HELP}")
        sys.exit(f"Ошибка Kaggle: {error}")
    except (OSError, ValueError) as error:
        sys.exit(f"Ошибка: {error}")

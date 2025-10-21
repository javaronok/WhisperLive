import os
import subprocess
import argparse
from importlib.util import spec_from_file_location, module_from_spec


def get_version_from_file(file_path: str) -> str:
    """Извлекает версию из файла version.py."""
    try:
        spec = spec_from_file_location("version", file_path)
        version_module = module_from_spec(spec)
        spec.loader.exec_module(version_module)
        return version_module.VERSION
    except Exception as e:
        raise RuntimeError(f"Не удалось получить версию из файла {file_path}: {e}")


def build_docker_image(image_name: str, version: str, dockerfile_path: str = "Dockerfile") -> None:
    """Собирает Docker образ."""
    try:
        subprocess.run(
            ["docker", "build", "-t", f"{image_name}:{version}", "-f", dockerfile_path, "."],
            check=True
        )
        print(f"Образ {image_name}:{version} успешно собран.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ошибка при сборке Docker образа: {e}")


def push_docker_image(image_name: str, publish_url: str, version: str) -> None:
    """Публикует Docker образ в реестр."""
    try:
        image_tag = f"{image_name}:{version}"
        publish_image_path = f"{publish_url}{image_tag}"
        subprocess.run(
            ["docker", "tag", image_tag, publish_image_path ],
            check=True
        )
        subprocess.run(
            ["docker", "push", publish_image_path],
            check=True
        )
        print(f"Образ {image_name}:{version} успешно опубликован.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ошибка при публикации Docker образа: {e}")


def main():
    # Получаем имя образа из переменных окружения
    image_name = os.getenv("DOCKER_IMAGE_NAME")
    if not image_name:
        raise ValueError("Переменная окружения DOCKER_IMAGE_NAME не установлена")

    parser = argparse.ArgumentParser(description=f"Сборка и публикация Docker образа {image_name}.")
    parser.add_argument(
        "--push",
        action="store_true",
        help="Публиковать образ после сборки"
    )
    args = parser.parse_args()

    app_version_path = os.getenv("APP_VERSION_SCRIPT_PATH", "./app/version.py")

    # Получаем версию из файла
    version = get_version_from_file(app_version_path)
    print(f"Версия из version.py: {version}")

    # Получаем файл сборки образа
    docker_file_path = os.getenv("DOCKER_FILE_PATH", "Dockerfile")
    print(f"Docker-скрипт сборки образа: {docker_file_path}")

    # Собираем образ
    build_docker_image(image_name, version, docker_file_path)

    # Публикуем образ, если указан флаг --push
    if args.push:
        publish_url = os.getenv("CR_PUBLISH_URL")
        if not image_name:
            raise ValueError("Переменная окружения CR_PUBLISH_URL не установлена")

        # Добавляем "/" в конце, если его нет
        if not publish_url.endswith('/'):
            publish_url += '/'

        push_docker_image(image_name, publish_url, version)


if __name__ == "__main__":
    main()

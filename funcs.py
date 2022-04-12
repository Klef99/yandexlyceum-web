import hashlib
import os


def pass_to_hash(password):
    """Функция конвертирует полученный пароль в хеш для хранения в базе данных"""
    salt = os.urandom(32)
    key = hashlib.pbkdf2_hmac(
        'sha256',  # Используемый алгоритм хеширования
        password.encode('utf-8'),  # Конвертируется пароль в байты
        salt,  # Предоставляется соль
        100000  # 100000 итераций SHA-256
    )
    return key.hex(), salt.hex()


def pass_check(password, key, salt):
    key = bytes.fromhex(key)
    salt = bytes.fromhex(salt)
    new_key = hashlib.pbkdf2_hmac(
        'sha256',  # Используемый алгоритм хеширования
        password.encode('utf-8'),  # Конвертируется пароль в байты
        salt,  # Предоставляется соль
        100000  # 100000 итераций SHA-256
    )
    if key == new_key:
        return True
    return False
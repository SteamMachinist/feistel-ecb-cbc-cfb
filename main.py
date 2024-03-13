import numpy as np


def get_t(bits):
    if bits == 32:
        return np.uint32
    elif bits == 64:
        return np.uint64
    else:
        raise ValueError("error")


def to_int(arr, bits):
    return np.packbits(arr).view(get_t(bits))[0]


def to_array(num, bits):
    return np.unpackbits(np.array([num], dtype=get_t(bits)).view(np.uint8))


def generate_keys(rounds):
    first = np.random.randint(2, size=64, dtype=np.uint8)
    first_int = to_int(first, 64)
    key = []
    for i in range(rounds):
        key.append(to_array(first_int << np.uint64(i * 3), 64))
    return np.array(key)


def f_function(data, k):
    """
    :param data: np.uint32
    :param k: np.uint64
    :return: np.uint32
    """
    one = data << np.uint32(9)
    two = k >> np.uint64(11)
    two = to_int(to_array(two, 64)[:32], 32)
    three = ~(two ^ data)
    return one ^ three


def split_message(message):
    return [message[i:i + 64] for i in range(0, len(message), 64)]


def complete_block(block):
    block_len = len(block)
    padded = 0
    if block_len < 64:
        padded = 64 - block_len
        block = list(block) + [0] * (64 - len(block))
    return block, padded


def get_blocks(message):
    blocks = split_message(message)
    last_block, completed = complete_block(blocks[-1])
    blocks[-1] = last_block
    blocks.insert(0, to_array(completed, 64))
    return blocks


def remove_completion(decoded_message):
    completed = to_int(decoded_message[:64], 64)
    decoded_message = decoded_message[64:]
    return decoded_message[:-int(completed)]


def encode_block(block, key):
    L, R = to_int(block[:32], 32), to_int(block[32:], 32)
    for k in key:
        temp = R
        F = f_function(R, k)
        R = F ^ L
        L = temp
    return np.append(to_array(R, 32), to_array(L, 32))


def decode_block(encoded_block, key):
    L, R = to_int(encoded_block[:32], 32), to_int(encoded_block[32:], 32)
    for k in reversed(key):
        temp = R
        F = f_function(R, k)
        R = F ^ L
        L = temp
    return np.append(to_array(R, 32), to_array(L, 32))


def ecb_encode(message, key, iv='unused'):
    encoded_message = []
    blocks = get_blocks(message)
    for block in blocks:
        encrypted_block = encode_block(block, key)
        encoded_message.extend(encrypted_block)
    return np.array(encoded_message)


def ecb_decode(encoded_message, key, iv='unused'):
    decoded_message = []
    for block in split_message(encoded_message):
        decoded_block = decode_block(block, key)
        decoded_message.extend(decoded_block)
    return np.array(remove_completion(decoded_message))


def cbc_encode(message, key, iv):
    encoded_message = []
    previous_block = iv
    blocks = get_blocks(message)
    for block in blocks:
        xor_result = to_array(to_int(block, 64) ^ to_int(previous_block, 64), 64)
        encoded_block = encode_block(xor_result, key)
        encoded_message.extend(encoded_block)
        previous_block = encoded_block
    return np.array(encoded_message)


def cbc_decode(encoded_message, key, iv):
    decoded_message = []
    previous_block = iv
    for block in split_message(encoded_message):
        decoded_block = decode_block(block, key)
        xor_result = to_array(to_int(decoded_block, 64) ^ to_int(previous_block, 64), 64)
        decoded_message.extend(xor_result)
        previous_block = block
    return np.array(remove_completion(decoded_message))


def cfb_encode(message, key, iv):
    encoded_message = []
    previous_block = iv
    blocks = get_blocks(message)
    for block in blocks:
        encoded_block = encode_block(previous_block, key)
        xor_result = to_array(to_int(block, 64) ^ to_int(encoded_block, 64), 64)
        encoded_message.extend(xor_result)
        previous_block = xor_result
    return np.array(encoded_message)


def cfb_decode(encoded_message, key, iv):
    decoded_message = []
    previous_block = iv
    for block in split_message(encoded_message):
        encoded_block = encode_block(previous_block, key)
        xor_result = to_array(to_int(block, 64) ^ to_int(encoded_block, 64), 64)
        decoded_message.extend(xor_result)
        previous_block = block
    return np.array(remove_completion(decoded_message))


rounds = 5

key = generate_keys(rounds)
key_str = '\n'.join([''.join(map(str, k)) for k in key])
print(f"Key:\n{key_str}\n")

iv = np.random.randint(2, size=64, dtype=np.uint8)
print(f"Initialization vector:\n{''.join(map(str, iv))}\n")

message_size = 92

message = np.random.randint(2, size=message_size, dtype=np.uint8)
print(f"Original message: {''.join(map(str, message))}")

encoded_message = ecb_encode(message, key, iv)
print(f"Encoded message:  {''.join(map(str, encoded_message))}")

decoded_message = ecb_decode(encoded_message, key, iv)
print(f"Decoded message:  {''.join(map(str, decoded_message))}")

print(f"Original == Decoded: {all(message == decoded_message)}")

def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
  x = b''
  for b in bytestring:
    x = x + bytes([b])

  return x.decode("utf-8")
decode_utf8_bytes_to_str_wrong("你好".encode("utf-8"))
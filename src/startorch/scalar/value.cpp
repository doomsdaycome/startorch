// #include "startorch/scalar/value.hpp"
//
// #include "startorch/common/types.hpp"
// #include "startorch/engine/bridge.hpp"
// #include "startorch/memory/buffer.hpp"
//
// namespace startorch {
//
// Value::Value()
//     : buffer_(Buffer::GetNull()),
//       type_(ScalarType::kUndefined),
//       system_data_(nullptr) {}
//
// Value::Value(Value&& other) noexcept
//     : buffer_(other.buffer_),
//       type_(other.type_),
//       system_data_(other.system_data_) {
//   other.buffer_ = Buffer::GetNull();
//   other.type_ = ScalarType::kUndefined;
//   other.system_data_ = nullptr;
// }
//
// Value::Value(const Value& other) : Value() {
//   if (other.IsNull()) return;
//
//   type_ = other.type_;
//   system_data_ = other.system_data_;
//   buffer_ = system_data_->NewBuffer(other.buffer_.GetBytes(),
//                                     other.buffer_.GetMallocType());
//
//   Bridge(*system_data_, *system_data_).Copy(buffer_, other.buffer_);
// }
//
// Value::Value(ScalarType type, System& system) : Value() {
//   if (type == ScalarType::kUndefined || system.IsNull()) return;
//
//   type_ = type;
//   system_data_ = &system;
// }
//
// Value::~Value();
//
// Value& Value::operator=(Value&& other) noexcept;
// Value& Value::operator=(const Value& other);
//
// }  // namespace startorch

use std::ffi;

use ash::{vk, Entry, Instance};

pub struct Debug {
    loader: ash::extensions::ext::DebugUtils,
    messenger: vk::DebugUtilsMessengerEXT,
}

impl Debug {
    pub fn init(entry: &Entry, instance: &Instance) -> Result<Debug, vk::Result> {
        let debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            )
            .pfn_user_callback(Some(vulkan_debug_utils_callback));
        let loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);
        let messenger = unsafe { loader.create_debug_utils_messenger(&debug_create_info, None)? };
        Ok(Debug { loader, messenger })
    }
}

impl Drop for Debug {
    fn drop(&mut self) {
        unsafe {
            self.loader
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

pub unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut ffi::c_void,
) -> vk::Bool32 {
    let message = ffi::CStr::from_ptr((*p_callback_data).p_message);
    let severity = format!("{:?}", message_severity).to_lowercase();
    let ty = format!("{:?}", message_type).to_lowercase();
    if severity == "info" {
        let msg = message.to_str().expect("An error occurred in Vulkan debug utils callback. What kind of not-String are you handing me?");
        if msg.contains("DEBUG-PRINTF") {
            let msg = msg
                .to_string()
                .replace("Validation Information: [ UNASSIGNED-DEBUG-PRINTF ]", "");
            println!("[Debug][printf] {:?}", msg);
        }
    } else {
        println!("[Debug][{}][{}] {:?}", severity, ty, message);
    }
    vk::FALSE
}

#  Copyright (c) 2022. Py-Star.com
#  All rights reserved.

# @File    : machine_code.py
# @Time    : 2022/5/16 18:45
# @Author  : Weilin.Gao
# @Email   : gaoweilin@casxm.cn
# @Project : cr_cd_gateway

import hashlib
import os
import pathlib
import platform
import subprocess

import wmi  # type: ignore

class ComputerCodeGenerator:
    SEP_GROUP = ";"
    SEP_FIELD = "_"

    def generate_machine_id(self, **kwargs):
        raise NotImplementedError()

    def run(self):
        return self.get_sha256(self.generate_machine_id())

    @staticmethod
    def get_sha256(string: str):
        """
        Compute the SHA256 signature of a string.
        """
        return hashlib.sha256(string.encode("utf-8")).hexdigest()

    def start_process(self, command):
        output = subprocess.check_output(command, **self.collect_subprocess_args(False))
        return output

    @staticmethod
    def collect_subprocess_args(include_stdout=True):
        if not hasattr(subprocess, 'STARTUPINFO'):
            si = None
            env = None
        else:
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            env = os.environ

        args = {}
        if include_stdout:
            args = {'stdout': subprocess.PIPE}

        args.update({
            'env': env,
            'startupinfo': si,
            'stdin': subprocess.PIPE,
            'stderr': subprocess.PIPE,
        })
        return args


class LinuxHostCodeGenerator(ComputerCodeGenerator):
    # noinspection PyBroadException
    @staticmethod
    def get_dbus_machine_id():
        try:
            with open("/etc/machine-id") as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            pass
        try:
            with open("/var/lib/dbus/machine-id") as f:
                return f.read().strip()
        except (FileNotFoundError, PermissionError):
            pass
        return ""

    @staticmethod
    def get_inodes():
        cls = LinuxHostCodeGenerator
        files = ["/bin", "/etc", "/lib", "/root", "/sbin", "/usr", "/var"]
        inodes = []
        for file in files:
            try:
                p_file = pathlib.Path(file)
                if not p_file.exists():
                    continue
                inodes.append(p_file.stat().st_ino)
            except PermissionError:
                continue
        return cls.SEP_FIELD.join([str(x) for x in inodes])

    def generate_machine_id(self, **kwargs):
        return self.get_dbus_machine_id() + self.get_inodes()


class MacHostCodeGenerator(ComputerCodeGenerator):

    def generate_machine_id(self, **kwargs):
        res = self.start_process(["system_profiler", "SPHardwareDataType"])
        return self.get_sha256(res[res.index("UUID"):].strip())


class WindowsHostCodeGenerator(ComputerCodeGenerator):
    def important_system_files(self):
        pass

    @staticmethod
    def get_serial_number_of_physical_disk(ctrl, drive_letter='C:'):
        logical_disk = ctrl.Win32_LogicalDisk(Caption=drive_letter)[0]
        partition = logical_disk.associators()[1]
        physical_disc = partition.associators()[0]
        return physical_disc.SerialNumber

    @staticmethod
    def get_cpu_id(ctrl):
        cls = WindowsHostCodeGenerator
        cpu_properties = ctrl.Win32_Processor()[0]
        res = getattr(cpu_properties, "UniqueId", None)
        if not res:  # If no UniqueID, use ProcessorID
            res = getattr(cpu_properties, "ProcessorId", None)
        if not res:  # If no ProcessorId, use Name
            res = getattr(cpu_properties, "Name", None)
        if not res:  # If no Name, use Manufacturer
            res = getattr(cpu_properties, "Manufacturer", None)
        # Add clock speed for extra security
        clock_speed = getattr(cpu_properties, "MaxClockSpeed", 0)
        return f"{res}{cls.SEP_FIELD}{clock_speed}"

    @staticmethod
    def get_bios_info(ctrl):
        cls = WindowsHostCodeGenerator
        bios_properties = ctrl.Win32_BIOS()[0]
        bios_keys = ("Manufacturer", "SerialNumber")
        return cls.SEP_FIELD.join(getattr(bios_properties, k, "") for k in bios_keys)

    @staticmethod
    def get_base_board_info(ctrl):
        cls = WindowsHostCodeGenerator
        base_properties = ctrl.Win32_BaseBoard()[0]
        base_keys = ("Manufacturer", "SerialNumber")
        return cls.SEP_FIELD.join(getattr(base_properties, k, "") for k in base_keys)

    @staticmethod
    def get_long_term_hardware_devices():
        cls = WindowsHostCodeGenerator
        ctrl = wmi.WMI()
        # # disk, ctrl.Win32_PhysicalMedia():
        # candidate.append(cls.get_serial_number_of_physical_disk(ctrl, os.getenv("SystemDrive", "C")))
        # cpu,bios,mother_board
        candidate = [cls.get_cpu_id(ctrl), cls.get_bios_info(ctrl), cls.get_base_board_info(ctrl)]

        return cls.SEP_GROUP.join(candidate)

    def generate_machine_id(self, **kwargs):
        v = kwargs.pop("v", 1)
        return self.get_sha256(
            self.SEP_GROUP.join([
                self.get_product_id(v),
                self.get_long_term_hardware_devices()
            ]))

    def get_product_id(self, v):
        output = self.start_process(["cmd.exe", "/C", "wmic", "csproduct", "get", "uuid"])
        if v == 1:
            return output.decode('utf-8')
        elif v == 2:
            raw_output = output.decode('utf-8')
            return raw_output[raw_output.index("UUID") + 4:].strip("\r\n\t ")
        else:
            raise ValueError("Version can be either 1 or 2.")


def code_generator_factory() -> ComputerCodeGenerator:
    platform_str = platform.platform().lower()
    if "windows" in platform_str:
        """
        Get a unique identifier for this device. If you want the machine code to be the same in .NET on Windows, you
        can set v=2. More information is available here: https://help.cryptolens.io/faq/index#machine-code-generation
        """
        return WindowsHostCodeGenerator()
    elif "mac" in platform_str or "darwin" in platform_str:
        return MacHostCodeGenerator()
    elif "linux" in platform_str:
        return LinuxHostCodeGenerator()
    else:
        return LinuxHostCodeGenerator()


def get_machine_code():
    n = 4  # len of group
    s = code_generator_factory().generate_machine_id(v=2)
    return "-".join(s[i:i + n] for i in range(0, len(s), n))


if __name__ == "__main__":
    print(">>> \n")
    print(f">>> {get_machine_code()}")



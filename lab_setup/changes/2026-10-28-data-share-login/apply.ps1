param([string]$Repo, [hashtable]$Secrets)
$ErrorActionPreference = "Stop"

# CredWrite in-process, not cmdkey /pass: a password on a command line is
# recorded by process auditing, and PowerShell 5.1 mangles one containing quotes.
Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class PytweezerCredential {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    struct CREDENTIAL {
        public int Flags;
        public int Type;
        public string TargetName;
        public string Comment;
        public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
        public int CredentialBlobSize;
        public IntPtr CredentialBlob;
        public int Persist;
        public int AttributeCount;
        public IntPtr Attributes;
        public string TargetAlias;
        public string UserName;
    }

    [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    static extern bool CredWriteW(ref CREDENTIAL credential, int flags);

    const int CRED_TYPE_DOMAIN_PASSWORD = 2;
    // What cmdkey /add uses.
    const int CRED_PERSIST_ENTERPRISE = 3;

    public static void WriteDomainPassword(string target, string user, string password) {
        IntPtr blob = Marshal.StringToCoTaskMemUni(password);
        try {
            CREDENTIAL credential = new CREDENTIAL();
            credential.Type = CRED_TYPE_DOMAIN_PASSWORD;
            credential.TargetName = target;
            credential.UserName = user;
            credential.CredentialBlob = blob;
            credential.CredentialBlobSize = password.Length * 2;
            credential.Persist = CRED_PERSIST_ENTERPRISE;
            if (!CredWriteW(ref credential, 0)) {
                throw new Win32Exception(Marshal.GetLastWin32Error());
            }
        } finally {
            Marshal.ZeroFreeCoTaskMemUnicode(blob);
        }
    }
}
"@

[PytweezerCredential]::WriteDomainPassword("10.59.3.1", "PH-BEAST\pytweezer-reader", $Secrets["READER_PASSWORD"])

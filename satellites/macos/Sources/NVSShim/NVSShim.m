#import "include/NVSShim.h"

BOOL NVSCatchException(void (NS_NOESCAPE ^block)(void), NSError **error) {
    @try {
        block();
        return YES;
    } @catch (NSException *exception) {
        if (error) {
            NSString *reason = exception.reason ?: exception.name;
            *error = [NSError errorWithDomain:@"NovaVoiceSatellite.ObjCException"
                                         code:-1
                                     userInfo:@{NSLocalizedDescriptionKey : reason}];
        }
        return NO;
    }
}

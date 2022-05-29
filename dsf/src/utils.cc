/*!
 *  Copyright (c) 2022 by Contributors
 */

#include <arpa/inet.h>
#include <dmlc/logging.h>
#include <netdb.h>
#include <netinet/ip.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstring>

namespace dgl {
namespace dsf {

int GetAvailablePort() {
  struct sockaddr_in addr;
  addr.sin_port = htons(0);   // 0 means let system pick up an available port.
  addr.sin_family = AF_INET;  // IPV4
  addr.sin_addr.s_addr = htonl(INADDR_ANY);  // set addr to any interface

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  if (0 != bind(sock, (struct sockaddr*)&addr, sizeof(struct sockaddr_in))) {
    DLOG(WARNING) << "bind()";
    return 0;
  }
  socklen_t addr_len = sizeof(struct sockaddr_in);
  if (0 != getsockname(sock, (struct sockaddr*)&addr, &addr_len)) {
    DLOG(WARNING) << "getsockname()";
    return 0;
  }

  int ret = ntohs(addr.sin_port);
  close(sock);
  return ret;
}

std::string GetHostName() {
  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);

  struct addrinfo hints = {0};
  hints.ai_family = AF_UNSPEC;
  hints.ai_flags = AI_CANONNAME;

  struct addrinfo* res = 0;
  std::string fqdn;
  if (getaddrinfo(hostname, 0, &hints, &res) == 0) {
    // The hostname was successfully resolved.
    fqdn = std::string(res->ai_canonname);
    freeaddrinfo(res);
  } else {
    // Not resolved, so fall back to hostname returned by OS.
    LOG(FATAL) << " ERROR: No HostName.";
  }
  return fqdn;
}

}  // namespace dsf
}  // namespace dgl
